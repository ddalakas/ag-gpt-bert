import random
import os

def split_jsonl(input_path, output_dir, ratios=(0.8, 0.1, 0.1), seed=0):
    """
    Splits a JSONL file into train/val/test according to given ratios.
    Writes three files: train.jsonl, val.jsonl, and test.jsonl to the output directory.
    """
    
    random.seed(seed)

    # Read all lines
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    total = len(lines)
    indices = list(range(total))

    random.shuffle(indices)

    # Compute split boundaries
    train_end = int(ratios[0] * total)
    val_end   = train_end + int(ratios[1] * total)

    splits = {
        'train': indices[:train_end],
        'val':   indices[train_end:val_end],
        'test':  indices[val_end:]
    }

    os.makedirs(output_dir, exist_ok=True)
    for split_name, idxs in splits.items():
        out_path = os.path.join(output_dir, f"{split_name}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as out_f:
            for i in idxs:
                out_f.write(lines[i])
        print(f"Wrote {len(idxs)} examples to {out_path}")

if __name__ == "__main__":
    split_jsonl(
        input_path="../corpus/ancient_greek_corpus.jsonl",
        output_dir="../splits/"
    )
