import os
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Processor,
    TrainingArguments,
    Trainer,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, Audio

# =========================
# CONFIG
# =========================
CONFIG = {
    "dataset_name": "stapesai/ssi-speech-emotion-recognition",
    "output_dir": "./wav2vec2-emotion-speech-recognition",
    "sample_rate": 16000,
    "max_duration": 6.0,
    "batch_size": 8,
    "epochs": 10,
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 2,
    "freeze_feature_encoder": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASET WRAPPER
# =========================
class HFEmotionDataset(Dataset):
    def __init__(self, dataset, label2id, processor, augment=False):
        self.dataset = dataset
        self.label2id = label2id
        self.processor = processor
        self.augment = augment
        self.max_samples = int(CONFIG["sample_rate"] * CONFIG["max_duration"])

    def __len__(self):
        return len(self.dataset)

    def augment_audio(self, x):
        if np.random.rand() < 0.5:
            x += 0.005 * np.random.randn(len(x))
        if np.random.rand() < 0.5:
            x *= np.random.uniform(0.7, 1.3)
        return x

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # The 'file_path' column is the Audio column in this dataset
        audio_array = item["file_path"]["array"]
        label_str = item["emotion"]
        
        # Ensure correct length / padding handled by processor mostly, but we can trim/pad here if needed
        # Just simple truncation for safety
        if len(audio_array) > self.max_samples:
            audio_array = audio_array[:self.max_samples]
            
        if self.augment:
            audio_array = self.augment_audio(audio_array)

        inputs = self.processor(
            audio_array,
            sampling_rate=CONFIG["sample_rate"],
            return_tensors="pt",
        )

        if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None:
             attention_mask = inputs.attention_mask.squeeze(0)
        else:
             attention_mask = torch.ones(inputs.input_values.shape[1], dtype=torch.long)

        return {
            "input_values": inputs.input_values.squeeze(0),
            "attention_mask": attention_mask,
            "labels": torch.tensor(self.label2id[label_str], dtype=torch.long),
        }

# =========================
# MODEL
# =========================
class Wav2Vec2Emotion(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        if hasattr(self.wav2vec, "gradient_checkpointing_disable"):
            self.wav2vec.gradient_checkpointing_disable()
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_labels),
        )

    def freeze_feature_encoder(self):
        self.wav2vec.feature_extractor._freeze_parameters()

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.wav2vec(input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state

        if attention_mask is not None:
            # Wav2Vec2 downsamples time, so convert sample-level mask to feature-level mask
            if hasattr(self.wav2vec, "_get_feature_vector_attention_mask"):
                feat_mask = self.wav2vec._get_feature_vector_attention_mask(
                    hidden.shape[1], attention_mask
                )
            else:
                feat_mask = attention_mask

            if feat_mask.shape[1] != hidden.shape[1]:
                feat_mask = torch.ones(
                    hidden.shape[:2], dtype=torch.long, device=hidden.device
                )

            mask = feat_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        else:
            pooled = torch.mean(hidden, dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

# =========================
# METRICS
# =========================
def compute_metrics(pred):
    y_true = pred.label_ids
    y_pred = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print(f"Loading dataset: {CONFIG['dataset_name']}...")
    ds = load_dataset(CONFIG["dataset_name"])
    
    # Ensure audio is 16kHz
    ds = ds.cast_column("file_path", Audio(sampling_rate=CONFIG["sample_rate"]))

    # Prepare Labels
    print("Preparing labels...")
    labels_list = sorted(list(set(ds["train"]["emotion"])))
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels_list)
    print(f"Labels: {labels_list}")

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")

    # Create Datasets
    # Use 'validation' split if available, else split train
    if "validation" in ds:
        val_split = ds["validation"]
        train_split = ds["train"]
    else:
        # Fallback split
        print("No validation split found. Splitting 'train'...")
        ds_split = ds["train"].train_test_split(test_size=0.2)
        train_split = ds_split["train"]
        val_split = ds_split["test"]
        
    test_split = ds["test"] if "test" in ds else val_split

    train_ds = HFEmotionDataset(train_split, label2id, processor, augment=True)
    val_ds = HFEmotionDataset(val_split, label2id, processor)
    test_ds = HFEmotionDataset(test_split, label2id, processor)

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Initialize Model
    model = Wav2Vec2Emotion(num_labels).to(device)
    if CONFIG.get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()

    # Training Arguments
    tb_log_dir = os.path.join(CONFIG["output_dir"], "runs")
    run_name = f"wav2vec2-emotion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        run_name=run_name,
        logging_dir=tb_log_dir,
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        num_train_epochs=CONFIG["epochs"],
        learning_rate=CONFIG["learning_rate"],
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        gradient_checkpointing=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    final_model_path = os.path.join(CONFIG["output_dir"], "model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    processor.save_pretrained(CONFIG["output_dir"])
    with open(os.path.join(CONFIG["output_dir"], "labels.json"), "w") as f:
        json.dump(label2id, f, indent=2)

    # Evaluate
    print("\n=== TEST SET EVALUATION ===")
    preds = trainer.predict(test_ds)
    y_pred = preds.predictions.argmax(-1)
    y_true = preds.label_ids

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=labels_list,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", xticklabels=labels_list, yticklabels=labels_list
    )
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "confusion_matrix.png"))
