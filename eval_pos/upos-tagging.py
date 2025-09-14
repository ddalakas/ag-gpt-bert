#!/usr/bin/env python
# coding: utf-8

"""
This script was adapted for offline wandb tracking and using a custom Ancient Greek BPE tokenizer
Credit is due to the authors at https://github.com/Heidelberg-NLP/

The original script can be found at:
https://github.com/Heidelberg-NLP/ancient-language-models/blob/main/src/ancient-language-models/pos_tagging.py

UPoS tagging script reading conllu data and a config.py file to perform a wandb run.
"""

from functools import partial
import sys
import logging
import warnings

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score

from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    Value,
    Features,
    Sequence,
)
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from utils import prepare_data
from transformers import PreTrainedTokenizerFast

import os
import wandb
import random
import torch
import numpy as np

os.environ["WANDB_MODE"] = "offline"

from transformers import Trainer

class SafeTrainer(Trainer):
    def save_model(self, output_dir: str, _internal_call: bool = False):
        # Always use safe_serialization=False
        self.model.save_pretrained(output_dir, safe_serialization=False)

def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes input text and aligns the labels with the tokenized output.
    """

    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=256
    )
    tokenized_labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        tokenized_labels.append(label_ids)
    tokenized_inputs["labels"] = tokenized_labels
    return tokenized_inputs


def pos_processor(sentence):
    """
    Processes a sentence to extract tokens and their corresponding PoS labels.
    """
    sent_tokens = [token["form"] for token in sentence]
    sent_labels = [token["upos"] for token in sentence]
    return sent_tokens, sent_labels

def compute_metrics(p):
    predictions, label_ids = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, label_ids)
    ]
    true_labels = [
        [labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, label_ids)
    ]

    precision = precision_score(true_labels, true_predictions)
    recall = recall_score(true_labels, true_predictions)
    f1 = f1_score(true_labels, true_predictions)
    accuracy = accuracy_score(true_labels, true_predictions)

    print(classification_report(true_labels, true_predictions))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

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

def train(config, hybrid_name, seed=42, task_name="upos"):
    """
    Method for finetuning on the training set, reporting validation statistics,
    and evaluationg on the test set after training.
    """
    # Remove mode=offline beneath for online tracking
    with wandb.init(config=config,mode="offline") as run:
        run_config = run.config

        tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizer/tokenizer.json",
                                        unk_token="<unk>",
                                        pad_token="<pad>",
                                        cls_token="<s>",
                                        sep_token="</s>",
                                        mask_token="<mask>")

        model_config = AutoConfig.from_pretrained(run_config.model_name_or_path, trust_remote_code=True)
        model_config.label2id = label2id
        model_config.id2label = id2label
        model_config.num_labels = len(id2label)
        model = AutoModelForTokenClassification.from_pretrained(
            run_config.model_name_or_path, config=model_config, ignore_mismatched_sizes=True, trust_remote_code=True
        )

        logging.info(
            "Loaded model %s. Number of parameters: %s.",
            run_config.model_name_or_path,
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )

        tokenized_datasets = datasets.map(
            partial(
                tokenize_and_align_labels,
                tokenizer=tokenizer,
            ),
            batched=True,
        )
        logging.info("Tokenized datasets: %s", tokenized_datasets)

        # Create an output directory for this particular seed
        seed_output_dir = os.path.join(output_dir, f"seed_{seed}")

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

warnings.filterwarnings(
    "ignore", message=".*seems not to be NE tag.*", category=UserWarning
)

# Parse command line arguments
if len(sys.argv) < 2:
    print(f"Usage: python upos-tagging.py upos-config.py [seed]")
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

train_sentences, train_labels = prepare_data(train_file, pos_processor)
valid_sentences, valid_labels = prepare_data(valid_file, pos_processor)
test_sentences, test_labels = prepare_data(test_file, pos_processor)

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

logging.info(
    "test: %s sentences, %s labels, example: %s -> %s",
    len(test_sentences),
    len(test_labels),
    test_sentences[0],
    test_labels[0],
)

labels = sorted(
    list({label for sentence in train_labels + valid_labels + test_labels for label in sentence})
)
logging.info("%s labels: %s", len(labels), labels)


features = Features(
    {
        "tokens": Sequence(Value("string")),
        "labels": Sequence(ClassLabel(names=labels)),
    }
)

train_dataset = Dataset.from_dict(
    {"tokens": train_sentences, "labels": train_labels}, features=features
)
valid_dataset = Dataset.from_dict(
    {"tokens": valid_sentences, "labels": valid_labels}, features=features
)
test_dataset = Dataset.from_dict(
    {"tokens": test_sentences, "labels": test_labels}, features=features
)

datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset, "test": test_dataset})
label2id = {label: i for i, label in enumerate(labels)}
id2label = dict(enumerate(labels))

if __name__ == "__main__":
    for lr in run_config["parameters"]["learning_rate"]["values"]:
        config = {k: (v["value"] if "value" in v else v["values"][0]) for k, v in run_config["parameters"].items()}
        config["learning_rate"] = lr
        hybrid_name = "Ratio_" + config["model_name_or_path"][-3:] # Extract the ratio from the model name
        task_name = "upos"
        
        # Run training with the specified seed
        final_accuracy = train(config=config, hybrid_name=hybrid_name, seed=seed, task_name=task_name)
        
        print(f"Training completed with seed {seed}. Final accuracy: {final_accuracy:.6f}")
        print(f"Results saved to: {hybrid_name}-{task_name}-accuracy.txt")