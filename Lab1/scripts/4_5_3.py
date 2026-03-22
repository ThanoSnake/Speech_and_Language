#!/usr/bin/env python3
"""
DNN-HMM training for USC-TIMIT phone recognition.
Run from ~/kaldi/egs/usc/ after sourcing path.sh:
    . ./path.sh && python3 scripts/4_5_3.py
"""

import os
import sys
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import kaldi_io
except ImportError:
    print("kaldi_io not found. Install with: pip install kaldi_io")
    sys.exit(1)


class TorchSpeechDataset(Dataset):
    """
    Frame-level dataset backed by Kaldi feature and alignment files.

    Properties:
        feats       : (N, D) float32 array of all frames concatenated
        labels      : (N,)   int64 array of pdf-id per frame
        uttids      : list of utterance IDs in order
        end_indices : list of cumulative end frame indices per utterance
    """

    def __init__(self, feat_scp, ali_dir, final_mdl):
        self.uttids = []
        self.end_indices = []

        # Read features
        feats_dict = {}
        for uttid, mat in kaldi_io.read_mat_scp(feat_scp):
            feats_dict[uttid] = mat

        # Read alignments: convert transition-ids → pdf-ids
        ali_cmd = (
            f'ali-to-pdf {final_mdl} '
            f'"ark:gunzip -c {ali_dir}/ali.*.gz |" ark:-'
        )
        labels_dict = {}
        for uttid, ali in kaldi_io.read_vec_int_ark(f'ark:{ali_cmd} |'):
            labels_dict[uttid] = ali

        # Merge, keeping only utterances present in both
        feat_list = []
        label_list = []
        cumulative = 0
        for uttid in sorted(feats_dict.keys()):
            if uttid not in labels_dict:
                continue
            f = feats_dict[uttid]
            l = labels_dict[uttid]
            n = min(len(f), len(l))
            feat_list.append(f[:n])
            label_list.append(l[:n])
            self.uttids.append(uttid)
            cumulative += n
            self.end_indices.append(cumulative)

        self.feats = np.concatenate(feat_list, axis=0).astype(np.float32)
        self.labels = np.concatenate(label_list, axis=0).astype(np.int64)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return torch.tensor(self.feats[idx]), torch.tensor(self.labels[idx])


class DNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return total_loss / total, correct / total


def export_posteriors(model, feat_scp, output_path, log_priors, device):
    """Write log pseudo-likelihoods log P(x|s) ≈ log P(s|x) − log P(s) as text ark."""
    model.eval()
    log_priors_t = torch.tensor(log_priors, dtype=torch.float32).to(device)
    with open(output_path, 'w') as f_out:
        with torch.no_grad():
            for uttid, feats in kaldi_io.read_mat_scp(feat_scp):
                feats_t = torch.tensor(feats, dtype=torch.float32).to(device)
                logits = model(feats_t)
                log_post = torch.log_softmax(logits, dim=1)
                log_like = (log_post - log_priors_t).cpu().numpy()
                f_out.write(f'{uttid}  [\n')
                for row in log_like:
                    f_out.write('  ' + '  '.join(f'{v:.6f}' for v in row) + '\n')
                f_out.write(']\n')


def get_num_pdfs(final_mdl):
    result = subprocess.run(['am-info', final_mdl], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if 'number of pdfs' in line:
            return int(line.strip().split()[-1])
    raise RuntimeError("Could not determine number of pdfs from am-info")


def main():
    tri1_dir      = 'exp/tri1'
    ali_train_dir = 'exp/tri1_ali_train'
    ali_dev_dir   = 'exp/tri1_ali_dev'
    feat_dir      = 'exp/dnn_feats'
    out_dir       = 'exp/dnn'
    os.makedirs(out_dir, exist_ok=True)

    final_mdl = f'{tri1_dir}/final.mdl'

    # Hyperparameters
    hidden_dims = [512, 512, 512]
    dropout     = 0.2
    lr          = 1e-3
    batch_size  = 256
    num_epochs  = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    print('Loading train data...')
    train_set = TorchSpeechDataset(f'{feat_dir}/train.scp', ali_train_dir, final_mdl)
    print(f'  {len(train_set.uttids)} utterances, {len(train_set)} frames')

    print('Loading dev data...')
    dev_set = TorchSpeechDataset(f'{feat_dir}/dev.scp', ali_dev_dir, final_mdl)
    print(f'  {len(dev_set.uttids)} utterances, {len(dev_set)} frames')

    num_pdfs  = get_num_pdfs(final_mdl)
    input_dim = train_set.feats.shape[1]
    print(f'Input dim: {input_dim}, Output classes (pdf-ids): {num_pdfs}')

    # Log priors from training label counts
    counts = np.bincount(train_set.labels, minlength=num_pdfs).astype(np.float64)
    counts = np.maximum(counts, 1)
    log_priors = np.log(counts / counts.sum()).astype(np.float32)
    np.save(f'{out_dir}/log_priors.npy', log_priors)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    dev_loader   = DataLoader(dev_set,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = DNN(input_dim, hidden_dims, num_pdfs, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    best_dev_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        dv_loss, dv_acc = run_epoch(model, dev_loader,   optimizer, criterion, device, train=False)
        scheduler.step(dv_loss)
        print(f'Epoch {epoch:2d}: train loss={tr_loss:.4f} acc={tr_acc:.4f} | '
              f'dev loss={dv_loss:.4f} acc={dv_acc:.4f}')
        if dv_loss < best_dev_loss:
            best_dev_loss = dv_loss
            torch.save(model.state_dict(), f'{out_dir}/best_model.pt')

    # Export posteriors for dev and test using best model
    model.load_state_dict(torch.load(f'{out_dir}/best_model.pt'))
    for split in ('dev', 'test'):
        out_path = f'{out_dir}/posteriors_{split}.ark'
        print(f'Exporting posteriors for {split} → {out_path}')
        export_posteriors(model, f'{feat_dir}/{split}.scp', out_path, log_priors, device)

    print('Done.')


if __name__ == '__main__':
    main()
