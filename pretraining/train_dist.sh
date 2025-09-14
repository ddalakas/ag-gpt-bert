echo "Job started at: $(date +%Y%m%d-%H:%M:%S:%3N)"
start=`date +%s`

# Set up distributed training environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355
export NCCL_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting 4-GPU distributed training"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

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
    --tokenizer_path="../tokenizer/tokenizer.json" \
    --output_dir="../checkpoints" \
    --name="RUN NAME" \
    --hybrid_numerator=2 \
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

end=`date +%s`
runtime=$(((end-start)/60))
echo "Runtime with 4 GPUs was $runtime minutes."