"""
Script for tokenizing .jsonl text files into .bin files using a Hugging Face tokenizer.

Adapted from: https://github.com/ltgoslo/gpt-bert/blob/main/corpus_tokenization/tokenize_corpus.py
"""

from tokenizers import Tokenizer
import json
import argparse
import torch
from tqdm import tqdm
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=Path, default="../splits", help="Folder with input .jsonl files")
    parser.add_argument("--train_file", type=Path, default="train.jsonl", help="Training data file")
    parser.add_argument("--valid_file", type=Path, default="val.jsonl", help="Validation data file")
    parser.add_argument("--test_file", type=Path, default="test.jsonl", help="Test data file")
    parser.add_argument("--tokenizer_folder", type=Path, default="../tokenizer", help="Folder containing tokenizer")
    parser.add_argument("--tokenizer_file", type=Path, default="tokenizer.json", help="Tokenizer filename")
    parser.add_argument("--output_train_path", type=Path, default="../bin/train_tokenized.bin", help="Output path for train bin file")
    parser.add_argument("--output_valid_path", type=Path, default="../bin/val_tokenized.bin", help="Output path for validation bin file")
    parser.add_argument("--output_test_path", type=Path, default="../bin/test_tokenized.bin", help="Output path for test bin file")
    return parser.parse_args()


def tokenize_text(tokenizer, text):
    """
    Convert a text string into a tensor of token IDs.
    """
    ids = tokenizer.encode(text.strip(), add_special_tokens=False).ids
    return torch.tensor(ids, dtype=torch.int16)


def tokenize_file(input_filename, output_filename, tokenizer):
    """
    Read a .jsonl file, tokenize each entry, and save the results to a .bin file.
    """
    tokenized_documents = []
    n_subwords = 0

    for line in tqdm(input_filename.open("rt")):
        document = json.loads(line)
        tokenized_document = tokenize_text(tokenizer, document)
        tokenized_documents.append(tokenized_document)
        n_subwords += len(tokenized_document)

    torch.save(tokenized_documents, output_filename)
    print(f"Tokenized {len(tokenized_documents)} documents with {n_subwords} subwords in total.")

if __name__ == "__main__":
    args = parse_args()
    tokenizer_path = args.tokenizer_folder / args.tokenizer_file
    tokenizer = Tokenizer.from_file(str(tokenizer_path)) # load the tokenizer

    # tokenize train file
    input_train_path = args.data_folder / args.train_file
    tokenize_file(input_train_path, args.output_train_path, tokenizer)

    # tokenize validation file (if provided)
    if args.valid_file is not None:
        input_valid_path = args.data_folder / args.valid_file
        tokenize_file(input_valid_path, args.output_valid_path, tokenizer)

    # tokenize test file (if provided)
    if args.test_file is not None:
        input_test_path = args.data_folder / args.test_file
        tokenize_file(input_test_path, args.output_test_path, tokenizer)
