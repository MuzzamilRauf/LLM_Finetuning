import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-3B"


def main():
    print("=== ENVIRONMENT ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    print("\n=== LOADING TOKENIZER ===")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded successfully.")
    print(f"Vocabulary size: {len(tokenizer)}")

    print("\n=== LOADING MODEL ===")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    print(f"Model parameters: {model.num_parameters():,}")

    print("\n=== TEST FORWARD PASS ===")

    prompt = "What is machine learning?"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    print("Forward pass successful.")
    print(f"Logits shape: {outputs.logits.shape}")

    print("\n=== MODEL CHECK PASSED ===")


if __name__ == "__main__":
    main()