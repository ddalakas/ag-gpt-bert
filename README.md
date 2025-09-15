  <h2 align="center"><b>GPT-BERT for Sample-Efficient Language Modelling of Ancient Greek</b></h2>
  <h3 align="center">Dimitrios Dalakas</h3>
  <p align="center">
    <b>Email:</b> dlkdim001@myuct.ac.za<br>
    <i>University of Cape Town</i>
  </p>

---

In this project, we train an Ancient Greek language model (LM) using the GPT-BERT architecture. This repository includes a curated Ancient Greek pretraining corpus, a custom Ancient Greek tokenizer, and all scripts required to pretrain a GPT-BERT model.

Additionally, the repository provides scripts, and data, to finetune the model on three downstream tasks using the UD Ancient Greek Perseus treebank:

- Universal PoS (UPoS) tagging
- Language-specific PoS (XPoS) tagging
- Unlabelled dependency parsing (UAS)

The pretraining corpus contains texts from the Perseus Digital Library and the First1KGreek Project.

Links:

- [First1KGreek Project](https://github.com/OpenGreekAndLatin/First1KGreek)
- [Perseus Digital Library - Canonical Greek Literature](https://github.com/PerseusDL/canonical-greekLit)

The texts are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Lastly, a script is provided for using a pretrained GPT-BERT model to generate Ancient Greek text.

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

The trained tokenizer's `tokenizer.json` file is provided in the `tokenizer` directory. The GPT-BERT pretraining script requires texts to be pre-tokenized and stored in a binary format.

To do this:

1. Navigate to the `tokenize_scripts` directory.
2. Run the `tokenize_corpus.py` script to produce a `.bin` file for each split of the corpus. The tokenized binary files will be saved in the `bin` folder.

> **Note**  
> Additional scripts are provided to train or test a tokenizer:
>
> - `tokenizer.py` — Train a new Ancient Greek tokenizer (Byte-Level BPE with RoBERTa post-processing).
> - `test_tokenizer.py` — Test the tokenizer.

## Pretraining

The pretraining script requires a multi-GPU setup. Our models were trained on four GPUs on the same node. This setup is reflected in the `train_dist.sh` shell script in the `pretraining` folder. To start training a model, run the `train_dist.sh` script after modifying any necessary parameters.

## Finetuning

The finetuning tasks are divided into two categories: PoS tagging and dependency parsing.  
 Accordingly, there are two directories for these downstream tasks: `eval_pos` and `eval_parsing`.

> **Note**
> After pretraining, the `.bin` model checkpoint must be converted into a Hugging Face–compatible format, specifically **SafeTensors**.  
> This can be done with the `convert_to_hf.py` script located in the `eval_pos` directory.  
> For detailed instructions, see the `README.md` files in `eval_pos` and/or `eval_parsing`.

## Text Generation

To generate Ancient Greek text using a trained GPT-BERT model, run the `generate_text.py` script. The maximum number of newly generated tokens can be specified, as well as the number of sequences. The default prompt and sampling parameters (such as `top_k`, `top_p`, and `temperature`) are set in the script but can be modified as needed.

# License

This project is licensed under [CC BY-NC-SA 2.5](https://creativecommons.org/licenses/by-nc-sa/2.5/).  
 See [`LICENSE`](LICENSE) for full legal code.
