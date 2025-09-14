# Log file
log_file = "logs/upos.log"

# Dataset files
train_file = "../finetuning_data/UD_Ancient_Greek-Perseus/grc_perseus-ud-train.conllu"
valid_file = "../finetuning_data/UD_Ancient_Greek-Perseus/grc_perseus-ud-dev.conllu"
test_file = "../finetuning_data/UD_Ancient_Greek-Perseus/grc_perseus-ud-test.conllu"

# Output directory for models
output_dir = "models/upos"

# Hyperparameter configuration
run_config = {
    "parameters": {
        "learning_rate": {"values": [1e-5]},  # Learning rate
        "model_name_or_path": {               # Hugging Face model folder path (replace with actual path)
            "values": ["PATH_TO_HUGGING_FACE_FOLDER"]
        },
        "num_train_epochs": {"values": [20]},  # Total epochs
        "per_device_train_batch_size": {"values": [16]},
        "per_device_eval_batch_size": {"values": [16]},
        "weight_decay": {"values": [0.01]}
    }
}
