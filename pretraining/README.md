# GPT-BERT Pretraining

This directory contains the code for pretraining a GPT-BERT model. The `train_distributed.py` file is used to train the model across multiple GPUs (4) on the same node. The distributed training is implemented using PyTorch's native `torchrun` utility.

## Training Script

An example usage of the training script is given below:

```bash
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py \
    --train_path="../bin/train_tokenized.bin" \
    --valid_path="../bin/val_tokenized.bin" \
    --config_file="../configs/config.json" \
    --tokenizer_path="../tokenizers/tokenizer.json" \
    --output_dir="../checkpoints" \
    --name="Run Name" \
    --hybrid_numerator=1 \
    --hybrid_denominator=4 \
    --seq_length=128 \
    --local_batch_size=32 \
    --global_batch_size=128 \
    --learning_rate=1e-3 \
    --max_steps=40000 \
    --optimizer="lamb" \
    --weight_decay=0.1 \
    --warmup_proportion=0.016 \
    --cooldown_proportion=0.016 \
    --mask_p_start=0.3 \
    --mask_p_end=0.15 \
    --mask_random_p=0.1 \
    --mask_keep_p=0.1 \
    --mixed_precision \
    --validate_every=1000 \
    --save_every=1000 \
    --seed=42
```

## Hybrid CLM-MLM Training

The code implements a hybrid CLM-MLM training approach:

- A fraction of GPUs train with MLM and the remaining GPUs train with CLM
- This fraction is controlled with `hybrid_numerator/hybrid_denominator`
- The `hybrid_numerator` parameter controls the MLM fraction out of the total

NB: The code requires the number of GPUs to be a multiple of the specified `hybrid_denominator` i.e. to train with a 1:3 causal-to-masked ratio, the number
of GPUs used must be a multiple of four.

## Validation and Monitoring

The following metrics are logged to wandb:

- Training loss/perplexity
- Validation loss/perplexity (MLM and CLM)
- Token prediction accuracy
- Gradient norms
- Learning rates
- Batch sizes and sequence lengths
- Masking probabilities

## Usage Requirements

### Prerequisites

The code requires (among other libraries) :

- HuggingFace Tokenizers
- PyTorch with CUDA support
- wandb for experiment tracking

### Pretraining Data Preparation

Training data should be preprocessed and tokenized before training. This can be done using
the `tokenize_corpus.py` file in the `tokenize_scripts` directory.

## Checkpoints

The training script saves:

- Regular model weights
- EMA model weights
- Full training state
- Optimizer and scheduler states
