<h2 align="center"><b>GPT-BERT for Sample-Efficient Language Modelling of Ancient Greek</b></h2>
<h3 align="center">Dimitrios Dalakas</h3>
<p align="center">
  <b>Email:</b> dlkdim001@myuct.ac.za<br>
  <i>University of Cape Town</i>
</p>

---

In this project, we train an Ancient Greek language model (LM) using the GPT-BERT architecture.  
This repository includes a curated Ancient Greek pretraining corpus, a custom Ancient Greek tokenizer, and all scripts required to pretrain a GPT-BERT model.

Additionally, the repository provides scripts, and data, to finetune the model on three downstream tasks using the UD Ancient Greek Perseus treebank:

- Universal PoS (UPoS) tagging
- Language-specific PoS (XPoS) tagging
- Unlabelled dependency parsing (UAS)

The pretraining corpus contains texts from the Perseus Digital Library and the First1KGreek Project.

Links:

- [First1KGreek Project](https://github.com/OpenGreekAndLatin/First1KGreek)
- [Perseus Digital Library - Canonical Greek Literature](https://github.com/PerseusDL/canonical-greekLit)

# Reproduce Our Results

## Setup

Install the required Python dependencies by running:

```bash
pip install -r requirements.txt
```

## Pretraining Corpus Splits

Firstly, the pretraining corpus must be split into three sets: training, validation, and testing.
This can be done using the `split_data.py` script in the `preprocess` folder.

## Tokenization

The GPT-BERT pretraining script requires the texts to be pre-tokenized in a binary format.  
Use the `tokenizer.json` file supplied in the `tokenizer` directory to tokenize the different corpus splits.

1. Navigate to the `tokenize_scripts` directory.
2. Run the `tokenize_corpus.py` script to produce a `.bin` file for each split of the corpus.  
   The tokenized binary files will be saved in the `bin` folder.

**Note:** We also provide scripts to train a tokenizer from scratch and to test it:

- `tokenizer.py` - Train a new Ancient Greek tokenizer (Byte-Level BPE with Roberta post-processing).
- `test_tokenizer.py` - Test the tokenizer.

## Pretraining

The pretraining script requires a multi-GPU setup. Our models were trained on four GPUs on the same node. This setup is reflected in the `train_dist.sh` shell script in the `pretraining` folder. To start training a model, run the `train_dist.sh` script after modifying any necessary parameters.

## Finetuning

The finetuning tasks are divided into two categories: PoS tagging and dependency parsing.  
Accordingly, there are two directories for these downstream tasks: `eval_pos` and `eval_parsing`. Refer to the `README.md` in each directory for detailed instructions.

**Note:** After pretraining, the `.bin` model checkpoint must be converted to a format supported by Hugging Face, specifically the SafeTensors format.
