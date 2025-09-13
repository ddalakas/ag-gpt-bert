"""
Byte-Level BPE RoBERTa-style Tokenizer training script.

This script trains a Byte-Level BPE tokenizer with Roberta-style special tokens and
post-processing, using Hugging Face's `tokenizers` library.

Adapted from:
https://github.com/ltgoslo/gpt-bert/blob/main/tokenizer_creation/create_tokenizer.py
"""

import argparse
import json
from collections import Counter
from tqdm import tqdm
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import RobertaProcessing


def initialize_tokenizer(args):
    start_of_text_symbol = "<s>"
    end_of_text_symbol = "</s>"
    unk_symbol = "<unk>"
    mask_symbol = "<mask>"
    pad_symbol = "<pad>"

    special_tokens = [unk_symbol, start_of_text_symbol, end_of_text_symbol, pad_symbol, mask_symbol]
    special_tokens += [f"<special_{i}>" for i in range(11)]

    tokenizer = Tokenizer(BPE(
        unk_token=unk_symbol,
        byte_fallback=False
    ))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC()
    ])

    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
    tokenizer.decoder = ByteLevelDecoder()

    tokenizer.post_processor = RobertaProcessing(
        sep=(end_of_text_symbol, 2),
        cls=(start_of_text_symbol, 1),
        add_prefix_space=True,
        trim_offsets=True
    )

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )

    return tokenizer, trainer


def calculate_stats(tokenizer, args):
    counter, n_words = Counter(), 0
    all_tokens = []
    for i, document in enumerate(open(f"{args.validation_path}")):
        text = json.loads(document).strip()
        if len(text) > 0:
            n_words += len(text.split())
            encoding = tokenizer.encode(text)
            tokens = encoding.tokens
            counter.update(tokens)
            all_tokens += tokens
            if i == 0:
                print("Example of tokenization:")
                print(text)
                print(tokenizer.decode(encoding.ids))
                for j in encoding.ids:
                    print(j, tokenizer.id_to_token(j))

    sorted_subwords = counter.most_common()
    n_subwords = sum(freq for _, freq in sorted_subwords)
    print(f"Average splits per word: {n_subwords / n_words:.3f}", flush=True)

    f_95 = sorted_subwords[len(sorted_subwords) * 95 // 100][1]

    print(f"F_{{95%}} is {f_95}\n")

    with open(f"{args.vocab_path[:-5]}_stats.txt", "w") as f:
        f.write(f"Vocabulary size: {args.vocab_size}\n")
        f.write(f"Average splits per word: {n_subwords / n_words:.3f}\n")
        f.write(f"F_{{95%}} is {f_95}\n")
        sorted_subwords_str = '\n\t'.join(f"{freq}: {subword}" for subword, freq in sorted_subwords)
        f.write(f"Sorted subwords:\n\t{sorted_subwords_str}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tokenizer creation')
    parser.add_argument('--input_path', type=str, default="../splits/train.jsonl", help="Specify the input filename")
    parser.add_argument('--validation_path', type=str, default="../splits/val.jsonl", help="Specify the validation filename")
    parser.add_argument('--vocab_path', type=str, default="../tokenizer/tokenizer.json", help="Specify the output filename")
    parser.add_argument('--vocab_size', type=int, default=2**15, help="Number of subwords in the trained tokenizer")
    parser.add_argument('--min_frequency', type=int, default=10, help="Minimal number of occurrences of every candidate subword")
    args = parser.parse_args()

    print("Initializing a BPE tokenizer", flush=True)
    tokenizer, trainer = initialize_tokenizer(args)

    print("Training the tokenizer", flush=True)

    def iterator(file_path: str):
        for line in tqdm(open(file_path, encoding="utf-8", errors="replace")):
            line = json.loads(line).strip()
            if len(line) > 0:
                yield line

    tokenizer.train_from_iterator(iterator(args.input_path), trainer)

    print("Saving the tokenizer", flush=True)
    tokenizer.save(args.vocab_path)

    print("Trying to load the tokenizer...")
    tokenizer = Tokenizer.from_file(args.vocab_path)
    print("Success!")

    calculate_stats(tokenizer, args)