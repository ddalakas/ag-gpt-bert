# Tokenization

This folder contains scripts for training a Byte-Level BPE tokenizer for Ancient Greek, as well as testing the tokenizer and tokenizing the pretraining corpus.

## Scripts

1. **Train the tokenizer**  
   Run `tokenizer.py` to train the tokenizer on the Ancient Greek corpus.

2. **Test the tokenizer**  
   Run `test_tokenizer.py` to see how the tokenizer splits a sample text into tokens.

3. **Tokenize the pretraining corpus**  
   Run `tokenize_corpus.py` to tokenize the full pretraining corpus and save the results as `.bin` files in the `.bin` folder.
