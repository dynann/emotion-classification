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
    RobertaTokenizer,
    RobertaModel
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
    "output_dir": "./multimodal-emotion-recognition",
    "sample_rate": 16000,
    "max_duration": 6.0,
    "batch_size": 4, # Reduced batch size largely due to duplicate models in VRAM
    "epochs": 10,
    "learning_rate": 1e-5,
    "gradient_accumulation_steps": 4,
    "freeze_feature_encoder": True,
    "text_model_name": "roberta-base",
    "audio_model_name": "facebook/wav2vec2-base",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATASET WRAPPER
# =========================
class MultimodalEmotionDataset(Dataset):
    def __init__(self, dataset, label2id, audio_processor, text_tokenizer, augment=False):
        self.dataset = dataset
        self.label2id = label2id
        self.audio_processor = audio_processor
        self.text_tokenizer = text_tokenizer
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
        
        # --- Audio Processing ---
        # The 'file_path' column is the Audio column in this dataset
        audio_array = item["file_path"]["array"]
            
        # Ensure correct length / truncation
        if len(audio_array) > self.max_samples:
            audio_array = audio_array[:self.max_samples]
            
        if self.augment:
            audio_array = self.augment_audio(audio_array)

        audio_inputs = self.audio_processor(
            audio_array,
            sampling_rate=CONFIG["sample_rate"],
            return_tensors="pt",
        )

        audio_input_values = audio_inputs.input_values.squeeze(0)
        
        # Create audio mask (1 for valid, 0 for padding - though here we haven't padded yet, collator typically handles it)
        if hasattr(audio_inputs, "attention_mask") and audio_inputs.attention_mask is not None:
             audio_attention_mask = audio_inputs.attention_mask.squeeze(0)
        else:
             audio_attention_mask = torch.ones(audio_input_values.shape[0], dtype=torch.long)

        # --- Text Processing ---
        text = item["text"]
        if text is None:
            text = "" # Handle missing text if any
            
        text_inputs = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = text_inputs.input_ids.squeeze(0)
        text_attention_mask = text_inputs.attention_mask.squeeze(0)

        label_str = item["emotion"]

        return {
            "input_values": audio_input_values,
            "audio_attention_mask": audio_attention_mask,
            "input_ids": input_ids,
            "attention_mask": text_attention_mask, # Standard name for transformers (text)
            "labels": torch.tensor(self.label2id[label_str], dtype=torch.long),
        }

# =========================
# MODEL
# =========================
class MultimodalEmotionModel(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        # Audio Encoder
        self.wav2vec = Wav2Vec2Model.from_pretrained(CONFIG["audio_model_name"])
        if hasattr(self.wav2vec, "gradient_checkpointing_disable"):
            self.wav2vec.gradient_checkpointing_disable()
            
        # Text Encoder
        self.roberta = RobertaModel.from_pretrained(CONFIG["text_model_name"])
        
        # Classifier
        # Audio dim (768) + Text dim (768) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size + self.roberta.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels),
        )

    def freeze_feature_encoder(self):
        self.wav2vec.feature_extractor._freeze_parameters()
        # Optionally freeze roberta embeddings if needed
        # for param in self.roberta.embeddings.parameters():
        #     param.requires_grad = False

    def forward(self, input_values, audio_attention_mask=None, input_ids=None, attention_mask=None, labels=None):
        # --- Audio Forward ---
        # Wav2Vec2 expects 'attention_mask' argument for its mask. We pass audio_attention_mask to it.
        audio_outputs = self.wav2vec(input_values, attention_mask=audio_attention_mask)
        audio_hidden = audio_outputs.last_hidden_state

        # Mean Pooling for Audio
        if audio_attention_mask is not None:
            # Wav2Vec2 downsamples time, so convert sample-level mask to feature-level mask
            if hasattr(self.wav2vec, "_get_feature_vector_attention_mask"):
                feat_mask = self.wav2vec._get_feature_vector_attention_mask(
                    audio_hidden.shape[1], audio_attention_mask
                )
            else:
                # Fallback if specific method not available
                scale = audio_hidden.shape[1] / audio_attention_mask.shape[1]
                feat_mask_len = (audio_attention_mask.sum(dim=1) * scale).long()
                feat_mask = torch.zeros(audio_hidden.shape[:2], device=audio_hidden.device)
                for i, l in enumerate(feat_mask_len):
                    feat_mask[i, :l] = 1
            
            mask = feat_mask.unsqueeze(-1).to(dtype=audio_hidden.dtype)
            denom = mask.sum(dim=1).clamp(min=1.0)
            audio_pooled = (audio_hidden * mask).sum(dim=1) / denom
        else:
            audio_pooled = torch.mean(audio_hidden, dim=1)

        # --- Text Forward ---
        text_outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation (first token)
        text_pooled = text_outputs.last_hidden_state[:, 0, :]

        # --- Concatenate ---
        # Shape: (Batch, 768 + 768)
        combined_features = torch.cat((audio_pooled, text_pooled), dim=1)

        # --- Classification ---
        logits = self.classifier(combined_features)

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
def data_collator(features):
    # Manually collate to handle padding for audio inputs_values which might differ in length
    # Text input_ids are already padded to max_length=128 in dataset, but audio is raw
    
    batch = {}
    
    # Text fields (already tensors)
    batch["input_ids"] = torch.stack([f["input_ids"] for f in features])
    batch["attention_mask"] = torch.stack([f["attention_mask"] for f in features])
    batch["labels"] = torch.stack([f["labels"] for f in features])
    
    # Audio fields
    # Pad audio input_values to longest in batch
    input_values = [f["input_values"] for f in features]
    # Simple pad sequence
    max_len = max([v.shape[0] for v in input_values])
    # Pad with 0.0 (silence)
    padded_values = torch.zeros(len(input_values), max_len)
    padded_mask = torch.zeros(len(input_values), max_len, dtype=torch.long)
    
    for i, v in enumerate(input_values):
        l = v.shape[0]
        padded_values[i, :l] = v
        padded_mask[i, :l] = 1 # 1 for valid, 0 for pad
        
    batch["input_values"] = padded_values
    batch["audio_attention_mask"] = padded_mask
    
    return batch

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

    # Initialize Processors
    audio_processor = Wav2Vec2Processor.from_pretrained(CONFIG["audio_model_name"])
    text_tokenizer = RobertaTokenizer.from_pretrained(CONFIG["text_model_name"])

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

    train_ds = MultimodalEmotionDataset(train_split, label2id, audio_processor, text_tokenizer, augment=True)
    val_ds = MultimodalEmotionDataset(val_split, label2id, audio_processor, text_tokenizer)
    test_ds = MultimodalEmotionDataset(test_split, label2id, audio_processor, text_tokenizer)

    print(f"Train size: {len(train_ds)}")
    print(f"Val size: {len(val_ds)}")
    print(f"Test size: {len(test_ds)}")

    # Initialize Model
    model = MultimodalEmotionModel(num_labels).to(device)
    if CONFIG.get("freeze_feature_encoder", True):
        model.freeze_feature_encoder()

    # Training Arguments
    tb_log_dir = os.path.join(CONFIG["output_dir"], "runs")
    run_name = f"multimodal-emotion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
        eval_steps=50,
        save_strategy="no",
        gradient_checkpointing=False,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=50,
        report_to=["tensorboard"],
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro",
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False, # Essential for passing non-standard args (audio_attention_mask)
        overwrite_output_dir=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save
    final_model_path = os.path.join(CONFIG["output_dir"], "model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")
    audio_processor.save_pretrained(CONFIG["output_dir"])
    text_tokenizer.save_pretrained(CONFIG["output_dir"])
    
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
