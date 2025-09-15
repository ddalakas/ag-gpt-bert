# Unlabelled Dependency Parsing

This directory contains code for finetuning a GPT-BERT model on the task of unlabelled dependency parsing (UAS).

A shell script (`run_uas.sh`) is provided to run **three finetuning experiments**, each with a different random seed.

The GPT-BERT model implementation (with Hugging Face wrappers) is provided in `modeling_ltgbert.py`, and the configuration is defined in `configuration_ltgbert.py`. Both files are taken/adapted from the official GPT-BERT Hugging Face [repository](https://huggingface.co/ltg/gpt-bert-babylm-base/tree/main).

Finetuning hyperparameters can be adjusted in the `uas-config.py` file.

> **Note:** To create a Hugging Face (HF) checkpoint (with a SafeTensors file) for the model, use `convert_to_hf.py`. Make sure to copy `modeling_ltgbert.py` and `configuration_ltgbert.py` into the same HF folder. Ensure that the `config.json` file in the HF folder contains the `AutoMap` for using the Hugging Face wrappers. The correct `config.json` file is in the `eval_pos` directory. Lastly, the path to the HF folder must be specified in `uas-config.py`
