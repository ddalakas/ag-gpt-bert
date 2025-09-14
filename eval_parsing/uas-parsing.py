#!/usr/bin/env python
# coding: utf-8

"""
This script was adapted for offline wandb tracking and using a custom Ancient Greek BPE tokenizer
Credit is due to the authors at https://github.com/Heidelberg-NLP/

The original script can be found at:
https://github.com/Heidelberg-NLP/ancient-language-models/blob/main/src/ancient-language-models/unlabeled_parsing.py

Unlabeled dependency parsing script reading a config.py file to perform a wandb run.
"""

from functools import partial
import sys
import logging

from datasets import (
    Dataset,
    DatasetDict,
    Value,
    Features,
    Sequence,
)
from sklearn.metrics import accuracy_score
import torch
from transformers import (
    AutoConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    PreTrainedTokenizerFast,
)
from utils import prepare_data
from models import DependencyGPTBertForTokenClassification

import os
import wandb
import random
import torch
import numpy as np

os.environ["WANDB_MODE"] = "offline"

class SafeTrainer(Trainer):
    def save_model(self, output_dir: str, _internal_call: bool = False):
        # Always use safe_serialization=False
        self.model.save_pretrained(output_dir, safe_serialization=False)

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes input text and aligns the dependency head labels with the tokenized output.
    Adjusts head IDs to point to the correct subword indices, mapping root tokens to <bos>.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
        return_special_tokens_mask=True,
    )

    batch_size = len(examples["tokens"])
    tokenized_head_ids = []

    for i in range(batch_size):
        head_ids = examples["labels"][i]

        word_ids = tokenized_inputs.word_ids(batch_index=i)
        word_idx_to_token_idx = {}
        previous_word_idx = None
        for token_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                word_idx_to_token_idx[word_idx] = token_idx
            previous_word_idx = word_idx

        bos_token_idx = 0
        adjusted_head_ids = []
        for word_idx, head_word_idx in enumerate(head_ids):
            token_idx = word_idx_to_token_idx.get(word_idx, None)
            if token_idx is None:
                continue
            if head_word_idx == 0:
                head_token_idx = bos_token_idx
            else:
                adjusted_head_word_idx = head_word_idx - 1
                head_token_idx = word_idx_to_token_idx[adjusted_head_word_idx]

            adjusted_head_ids.append((token_idx, head_token_idx))

        labels = [-100] * len(word_ids)
        for token_idx, head_token_idx in adjusted_head_ids:
            labels[token_idx] = head_token_idx

        tokenized_head_ids.append(labels)

    tokenized_inputs["labels"] = tokenized_head_ids
    return tokenized_inputs


def compute_metrics(eval_pred):
    """
    Computes metrics, ignoring padded positions.
    """
    logits, labels = eval_pred
    max_seq_length = logits.shape[1]
    if labels.shape[1] < max_seq_length:

        padding_size = max_seq_length - labels.shape[1]

        labels = torch.nn.functional.pad(

            torch.tensor(labels), (0, padding_size), value=-100

        )
    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)
    mask = labels != -100
    valid_predictions = predictions[mask]
    valid_labels = labels[mask]
    accuracy = accuracy_score(valid_labels, valid_predictions)
    return {"accuracy": accuracy}


def head_processor(sentence):
    """
    Processes a sentence to extract tokens and their corresponding heads.
    """
    sent_tokens = [token["form"] for token in sentence]
    sent_labels = [token["head"] for token in sentence]
    return sent_tokens, sent_labels

def preprocess_logits_for_metrics(logits, labels):
    """
    Pads logits to ensure consistent shape across batch and sequence dimensions.
    """
    max_seq_length = 512
    current_seq_length = logits.shape[1]
    if current_seq_length < max_seq_length:
        padding_size = max_seq_length - current_seq_length
        logits = torch.nn.functional.pad(
            logits, (0, padding_size, 0, padding_size), value=float("-inf")
        )
    return logits


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def append_accuracy_to_file(accuracy, output_path, hybrid_name, task_name, seed):
    """
    Append this run's accuracy value on the test set to the output file.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Create filename with the hybrid variant's name
    filename = f"{hybrid_name}-{task_name}-accuracy.txt"
    filepath = os.path.join(output_path, filename)
 
    # Write the model's test accuracy to file
    with open(filepath, 'a') as f:
        f.write(f"{accuracy:.6f}\n")
 
    print(f"Appended accuracy of: {accuracy:.6f} to {filepath}, for seed={seed}")


def train(config, hybrid_name, seed=42, task_name="uas"):
    """
    Method for performing finetuning on training set, reporting validation statistics, and a test evaluation after training.
    """
    # Remove mode=offline beneath for online tracking
    with wandb.init(config=config, mode="offline") as run:
        run_config = run.config

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="../tokenizer/tokenizer.json",
            unk_token="<unk>",
            pad_token="<pad>",
            cls_token="<s>",
            sep_token="</s>",
            mask_token="<mask>"
        )

        model_config = AutoConfig.from_pretrained(run_config.model_name_or_path, trust_remote_code=True)
        model = DependencyGPTBertForTokenClassification.from_pretrained(
            run_config.model_name_or_path,
            config=model_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )

        logging.info(
            "Loaded model %s. Number of parameters: %s.",
            run_config.model_name_or_path,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        tokenized_datasets = datasets.map(
            partial(tokenize_and_align_labels, tokenizer=tokenizer),
            batched=True,
        )
        logging.info("Tokenized_datasets: %s", tokenized_datasets)

        # Create an output directory for this particular seed
        seed_output_dir = os.path.join(output_dir, f"{hybrid_name}-seed{seed}")

        training_arguments = {
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "per_device_train_batch_size": run_config.per_device_train_batch_size,
            "per_device_eval_batch_size": run_config.per_device_eval_batch_size,
            "num_train_epochs": run_config.num_train_epochs,
            "weight_decay": run_config.weight_decay,
            "push_to_hub": False,
            "metric_for_best_model": "accuracy",
            "load_best_model_at_end": True,
            "output_dir": seed_output_dir,
            "run_name": f"{task_name}_seed_{seed}",
            "report_to": "wandb",
            "learning_rate": run_config.learning_rate,
            "save_total_limit": 3,
            "seed": seed, 
        }
        args = TrainingArguments(**training_arguments)
        data_collator = DataCollatorForTokenClassification(tokenizer)
        trainer = SafeTrainer(
            model,
            args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.train() # Training

        evaluation = trainer.evaluate() # Validation
        logging.info("Validation evaluation: %s", evaluation)

        test_eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"]) # Test
        logging.info("Test evaluation results: %s", test_eval_results)

        trainer.model.save_pretrained(seed_output_dir, safe_serialization=False)

        # Extract accuracy and save to file
        final_accuracy = test_eval_results['eval_accuracy']
        append_accuracy_to_file(final_accuracy, output_dir, hybrid_name, task_name, seed)
        
        return final_accuracy

# Parse command line arguments
if len(sys.argv) < 2:
    print(f"Usage: python uas-parsing.py uas-config.py [seed]")
    sys.exit(1)
 
config_file = sys.argv[1]
 
# Get seed (default to 42)
if len(sys.argv) > 2:
    seed = int(sys.argv[2])
else:
    seed = 42
 
# Set seed for reproducibility
set_seed(seed)

# Load and execute config
with open(config_file, encoding="utf-8") as f:
    cfg = f.read()
    print(cfg)
    exec(cfg)
logging.basicConfig(filename=log_file, filemode="w", level=logging.INFO)

# Uncomment beneath for online tracking
#wandb.login()

train_sentences, train_labels = prepare_data(train_file, head_processor)
valid_sentences, valid_labels = prepare_data(valid_file, head_processor)
test_sentences, test_labels = prepare_data(test_file, head_processor)

logging.info(
    "train: %s sentences, %s labels, example: %s -> %s",
    len(train_sentences),
    len(train_labels),
    train_sentences[0],
    train_labels[0],
)
logging.info(
    "valid: %s sentences, %s labels, example: %s -> %s",
    len(valid_sentences),
    len(valid_labels),
    valid_sentences[0],
    valid_labels[0],
)

features = Features(
    {
        "tokens": Sequence(Value("string")),
        "labels": Sequence(Value("uint32")),
    }
)

train_dataset = Dataset.from_dict({"tokens": train_sentences, "labels": train_labels}, features=features)
valid_dataset = Dataset.from_dict({"tokens": valid_sentences, "labels": valid_labels}, features=features)
test_dataset = Dataset.from_dict({"tokens": test_sentences, "labels": test_labels}, features=features)

datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})

if __name__ == "__main__":
    for lr in run_config["parameters"]["learning_rate"]["values"]:
        config = {k: (v["value"] if "value" in v else v["values"][0]) for k, v in run_config["parameters"].items()}
        config["learning_rate"] = lr
        hybrid_name = "Ratio_" + config["model_name_or_path"][-3:] # Extract the ratio from the model name
        task_name = "uas"
        
        # Run training with the specified seed
        final_accuracy = train(config=config, hybrid_name=hybrid_name, seed=seed, task_name=task_name)
        
        print(f"Training completed with seed {seed}. Final accuracy: {final_accuracy:.6f}")
        print(f"Results saved to: {hybrid_name}-{task_name}-accuracy.txt")