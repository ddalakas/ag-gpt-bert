# Corpus Splits

This folder should contain the `.jsonl` files for each dataset split:

- **Training set**: `train.jsonl`
- **Validation set**: `val.jsonl`
- **Test set**: `test.jsonl`

The splits can be produced by running `split_data.py` in the `preprocess` folder.
Each file must contain one JSON string per line.
