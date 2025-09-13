"""
Script to load a trained tokenizer, tokenize a snippet from the test set,
and print the original and tokenized texts.
"""

from tokenizers import Tokenizer
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default="../splits/test.jsonl")
args = parser.parse_args()

# Load the trained tokenizer
tokenizer = Tokenizer.from_file('../tokenizer/tokenizer.json')

def test(text):
    """Tokenize a string and print each subtoken."""
    enc = tokenizer.encode(text)
    subtokens = [tokenizer.decode([tid], skip_special_tokens=True) for tid in enc.ids] # decode each token ID to inspect the individual BPE subwords
    print(" ".join(f"[{t}]" for t in subtokens)) # print subtokens with brackets for clarity

# Run on the first JSONL line of the test file
with open(args.test_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            text = json.loads(line)
            print("INPUT:", text)
            test(text)
            break