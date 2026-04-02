import os
import sys
import json
import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.load_datasets import load_MR

# ============================================================
# Config
# ============================================================
MODEL_NAME = "textattack/bert-base-uncased-SST-2"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_max")
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
MAX_LENGTH = 128
WARMUP_RATIO = 0.1
FGM_EPSILON = 0.5
MAX_EPOCHS = 10
PATIENCE = 3
SEEDS = [42, 123, 456]

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        "query", "key", "value",
        "attention.output.dense",
        "intermediate.dense", "output.dense",
    ],
    modules_to_save=["classifier"],
    bias="none",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Data
# ============================================================
X_train, y_train, X_test, y_test = load_MR()

le = LabelEncoder()
le.fit(list(set(y_train)))
y_train = le.transform(y_train)
y_test = le.transform(y_test)
n_classes = len(le.classes_)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(examples):
    return tokenizer(
        examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH
    )


def prepare_dataset(X, y):
    return Dataset.from_dict({"text": list(X), "label": list(y)})


train_set = prepare_dataset(X_train, y_train).map(tokenize, batched=True)
test_set = prepare_dataset(X_test, y_test).map(tokenize, batched=True)


# ============================================================
# Metrics
# ============================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "recall": recall_score(labels, preds, average="binary"),
        "f1": f1_score(labels, preds, average="binary"),
    }


# ============================================================
# FGM adversarial training
# ============================================================
class FGM:
    """Fast Gradient Method — perturb word embeddings along the gradient."""

    def __init__(self, model, epsilon=0.5):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and "word_embeddings" in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(self.epsilon * param.grad / norm)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class AdvTrainer(Trainer):
    """Trainer with FGM adversarial training on each step."""

    def __init__(self, fgm_epsilon=0.5, **kwargs):
        super().__init__(**kwargs)
        self.fgm = FGM(self.model, epsilon=fgm_epsilon)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Normal forward + backward
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(loss)

        # Adversarial forward + backward (gradients accumulate)
        self.fgm.attack()
        with self.compute_loss_context_manager():
            adv_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        self.accelerator.backward(adv_loss)
        self.fgm.restore()

        return loss.detach() / self.args.gradient_accumulation_steps


# ============================================================
# Train single model
# ============================================================
def train_single(seed):
    print(f"\n{'=' * 60}")
    print(f"  Training with seed {seed}")
    print(f"{'=' * 60}")

    set_seed(seed)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=n_classes
    )
    model = get_peft_model(base_model, LORA_CONFIG)

    if seed == SEEDS[0]:
        model.print_trainable_parameters()

    seed_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}")

    args = TrainingArguments(
        output_dir=seed_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=MAX_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
        logging_steps=50,
    )

    trainer = AdvTrainer(
        fgm_epsilon=FGM_EPSILON,
        model=model,
        args=args,
        train_dataset=train_set,
        eval_dataset=test_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()
    results = trainer.evaluate()

    # Save best model for ensemble
    trainer.save_model(os.path.join(seed_dir, "best"))

    print(f"  -> accuracy={results['eval_accuracy']:.4f}  f1={results['eval_f1']:.4f}")

    del model, base_model, trainer
    torch.cuda.empty_cache()

    return results


# ============================================================
# Ensemble
# ============================================================
def ensemble_evaluate():
    print(f"\n{'=' * 60}")
    print("  Ensemble evaluation")
    print(f"{'=' * 60}")

    all_logits = []
    for seed in SEEDS:
        best_dir = os.path.join(OUTPUT_DIR, f"seed_{seed}", "best")
        model = AutoModelForSequenceClassification.from_pretrained(best_dir)

        tmp_args = TrainingArguments(
            output_dir=os.path.join(OUTPUT_DIR, "tmp"),
            per_device_eval_batch_size=64,
        )
        tmp_trainer = Trainer(model=model, args=tmp_args)
        preds = tmp_trainer.predict(test_set)
        all_logits.append(preds.predictions)

        del model, tmp_trainer
        torch.cuda.empty_cache()

    avg_logits = np.mean(all_logits, axis=0)
    final_preds = np.argmax(avg_logits, axis=-1)
    labels = np.array(test_set["label"])

    results = {
        "accuracy": accuracy_score(labels, final_preds),
        "recall": recall_score(labels, final_preds, average="binary"),
        "f1": f1_score(labels, final_preds, average="binary"),
    }

    print(f"\n  Ensemble of {len(SEEDS)} models:")
    for k, v in results.items():
        print(f"    {k}: {v:.4f}")

    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    all_results = {}
    for seed in SEEDS:
        all_results[seed] = train_single(seed)

    ensemble_results = ensemble_evaluate()

    output = {
        "individual": {str(s): r for s, r in all_results.items()},
        "ensemble": ensemble_results,
        "config": {
            "model": MODEL_NAME,
            "learning_rate": LEARNING_RATE,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "fgm_epsilon": FGM_EPSILON,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "warmup_ratio": WARMUP_RATIO,
            "seeds": SEEDS,
        },
    }
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {results_path}")
