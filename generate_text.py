from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch
import argparse

def generate_text(model_name, checkpoint_file, prompt, max_new_tokens=20, num_return_sequences=5):
    print("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file="tokenizers/tokenizer.json",
        unk_token="<unk>",
        pad_token="<pad>",
        cls_token="<s>",
        sep_token="</s>",
        eos_token="</s>",
    )

    print(f"Loading model: {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading weights from {checkpoint_file} ...")
    try:
        state_dict = torch.load(checkpoint_file, map_location=torch.device('cpu'))

        # Remap 'head.*' to 'classifier.*' for mismatched weight names
        mapped_state_dict = {
            (k.replace("head.", "classifier.") if k.startswith("head.") else k): v
            for k, v in state_dict.items()
        }

        # Load raw weights from .bin checkpoint
        missing_keys, unexpected_keys = model.load_state_dict(mapped_state_dict, strict=False) 

        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")

    except Exception as e:
        print(f"Could not load weights: {e}")

    model.eval()

    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids
    print(f"\nPrompt: {tokenizer.decode(input_ids[0])}")

    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
            do_sample=True,     # Enable sampling for diverse outputs
            top_k=50,           # Consider only top 50 most likely tokens at each step
            top_p=0.95,         # Keeps text coherent by ignoring very unlikely words, but allows some variation
            temperature=1.0,    # Neutral randomness for sampling
            use_cache=False,    # Caching not supported in the GPT-BERT Hugging Face wrappers
        )

    print(f"\nGenerated sequences:")
    for i, sequence in enumerate(outputs):
        text = tokenizer.decode(sequence, skip_special_tokens=True)
        print(f"{i+1}. {text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Τῆς παιδείας ἔφη τὰς μὲν ῥίζας εἶναι πικράς, τὸν δὲ καρπὸν γλυκύν.", help="Prompt text")
    parser.add_argument("--ratio", type=str, default="2_4", help="Hybrid Ratio (0_4, 1_4, 2_4, 3_4, 4_4)")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--num_return_sequences", type=int, default=5)
    parser.add_argument("--hf_folder_name", type=str, required=True, help="Hugging Face folder for the model.")
    parser.add_argument("--checkpoint_name", type=str, required=True, help="Model checkpoint name (.bin file).")
    args = parser.parse_args()

    if args.checkpoint_name is None:
        args.checkpoint_name = f"checkpoints/{args.hf_folder_name}_{args.ratio}.bin"

    # Generate Ancient Greek text with specified parameters
    generate_text(
        model_name=args.hf_folder_name,
        checkpoint_file=args.checkpoint_name,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences
    )