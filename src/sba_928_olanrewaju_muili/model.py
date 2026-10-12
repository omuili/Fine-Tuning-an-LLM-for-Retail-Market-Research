"""Requirement 3: model loading, generation and fine-tuning."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from . import config


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(
    model_name: str, device: torch.device
) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Download (or load from cache) the pretrained tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model


class RetailDataset(Dataset):
    """Tokenized prompt/answer pairs for sequence-to-sequence training."""

    def __init__(self, examples: Sequence[Dict[str, str]], tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        inputs = self.tokenizer(
            example["prompt"],
            truncation=True,
            max_length=config.MAX_INPUT_LENGTH,
        )
        targets = self.tokenizer(
            text_target=example["answer"],
            truncation=True,
            max_length=config.MAX_TARGET_LENGTH,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs


def generate_answer(model, tokenizer, prompt: str, device: torch.device) -> str:
    """Generate a deterministic (beam search) answer for one prompt."""
    model.eval()
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config.MAX_INPUT_LENGTH,
    ).to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.MAX_TARGET_LENGTH,
            num_beams=2,
            do_sample=False,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def predict(model, tokenizer, examples, device: torch.device) -> List[str]:
    return [generate_answer(model, tokenizer, e["prompt"], device) for e in examples]


def fine_tune(
    model,
    tokenizer,
    examples: Sequence[Dict[str, str]],
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> List[Dict[str, float]]:
    """Fine-tune the model in place and return the average loss per epoch."""
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )
    loader = DataLoader(
        RetailDataset(examples, tokenizer),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_loss = epoch_loss / len(loader)
        losses.append({"epoch": epoch, "average_loss": average_loss})
        print(f"Epoch {epoch}/{epochs} - Loss: {average_loss:.4f}")

    return losses
