# PoS Tagging

This directory contains code for finetuning a GPT-BERT model on the task of Part-of-Speech (PoS) tagging.  
PoS tagging is divided into two tasks: Universal (UPoS) tagging and language-specific (XPoS) tagging.

A shell script is provided for each task to run **three finetuning experiments**, each initialized with a different random seed:

- **UPoS tagging** → `run_upos.sh`
- **XPoS tagging** → `run_xpos.sh`

The GPT-BERT model implementation (with Hugging Face wrappers) is provided in `modeling_ltgbert.py`, and the configuration is defined in `configuration_ltgbert.py`. Both files are taken/adapted from the official GPT-BERT Hugging Face [repository](https://huggingface.co/ltg/gpt-bert-babylm-base/tree/main).

Finetuning hyperparameters can be adjusted in the `upos-config.py` and `xpos-config.py` files, respectively.

**NB:** The path to the Hugging Face folder containing the model checkpoints must be specified in both `upos-config.py` and `xpos-config.py`. If you haven’t already created a Hugging Face checkpoint folder for the model, use `convert_to_hf.py` to do so. Make sure to copy `modeling_ltgbert.py` and `configuration_ltgbert.py`
into the HF checkpoint folder. Lastly, ensure the `config.json` file in the HF checkpoint contains the
'AutoMap' for the Hugging Face wrappers. The correct `config.json` file is in this directory.
