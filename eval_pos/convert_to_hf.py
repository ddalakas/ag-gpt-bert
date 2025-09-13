import argparse
import torch
from modeling_ltgbert import LtgbertForTokenClassification  # Contains GPT-BERT model with HF wrappers
from configuration_ltgbert import LtgbertConfig  # Config used for training the GPT-BERT model

def main():
    parser = argparse.ArgumentParser(description="Convert GPT-BERT .bin checkpoint to Hugging Face format.")
    parser.add_argument(
        "--config_dir",
        type=str,
        default="../configs",
        help="Directory containing the GPT-BERT training configuration (folder with config.json)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/pytorch_model.bin",  # default path to checkpoint .bin file
        help="Path to the .bin checkpoint file to load."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../hf_gpt-bert_model",  # default output folder name
        help="Directory to save the converted Hugging Face model."
    )
    args = parser.parse_args()

    model = LtgbertForTokenClassification(LtgbertConfig.from_pretrained(args.config_dir))

    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.save_pretrained(args.output_dir)
    print(f"Model successfully converted and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
