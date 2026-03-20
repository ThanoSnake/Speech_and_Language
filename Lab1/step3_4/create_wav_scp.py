import os

folders = ['train', 'test', 'dev']

base_path = "/home/feida/kaldi/egs/usc/data/wav/"

def create_wav_scp():
    for folder in folders:
        uttids_path = os.path.join(folder, 'uttids')
        wav_scp_path = os.path.join(folder, 'wav.scp')
        
        if not os.path.exists(uttids_path):
            print(f"Σφάλμα: Δεν βρέθηκε το {uttids_path}")
            continue
            
        with open(uttids_path, 'r') as f_in, open(wav_scp_path, 'w') as f_out:
            for line in f_in:
                utt_id = line.strip()
                if not utt_id:
                    continue
    
                wav_path = os.path.join(base_path, f"{utt_id}.wav")      
        
                f_out.write(f"{utt_id} {wav_path}\n")   

create_wav_scp()