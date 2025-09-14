# Unlabelled Dependency Parsing

This directory contains code for finetuning a GPT-BERT model on the task of unlabelled dependency parsing (UAS). A shell script is included to run three finetuning experiments, each with a different random seed.

The GPT-BERT model implementation (with Hugging Face wrappers) is provided in `modeling_ltgbert.py`, and the configuration is defined in `configuration_ltgbert.py`. Both files are taken/adapted from the official GPT-BERT Hugging Face [repository](https://huggingface.co/ltg/gpt-bert-babylm-base/tree/main).

Finetuning hyperparameters can be adjusted in the `uas-config.py` file.

**Note:** The path to the Hugging Face folder containing the model checkpoint must be specified in `uas-config.py`. If you haven’t already created a Hugging Face checkpoint folder for the model, use `convert_to_hf.py` in the `eval_pos` directory to do so. Make sure to copy `modeling_ltgbert.py` and `configuration_ltgbert.py`
into the HF checkpoint folder. Lastly, ensure the `config.json` file in the HF checkpoint contains the
'AutoMap' for the Hugging Face wrappers. The correct `config.json` file is in the `eval_pos` directory.
