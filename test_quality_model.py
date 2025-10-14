"""
Test the quality-trained model with its correct tokenizer
"""

import torch
from pathlib import Path
from models import IteraLiteModel
from utils.data import SimpleTokenizer

def load_quality_model():
    """Load the quality trained model with correct tokenizer"""
    # Load checkpoint
    checkpoint = torch.load('checkpoints/itera_lite_quality_best.pt', map_location='cpu')

    # Load tokenizer
    tokenizer = SimpleTokenizer(vocab_size=8000, level='word')
    tokenizer.load('data/tokenizer_quality.json')

    # Create model
    from models.config import IteraLiteConfig
    config_dict = checkpoint['config']
    config = IteraLiteConfig(**config_dict)

    model = IteraLiteModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, tokenizer, config

def generate(model, tokenizer, prompt, max_length=50, temperature=1.0):
    """Generate text"""
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.special_tokens.get('<BOS>', 0)]

    input_ids = torch.tensor([tokens], dtype=torch.long)

    generated = tokens.copy()

    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _, _ = model(input_ids)
            next_token_logits = logits[0, -1, :] / temperature

            # Sample
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            # Stop at EOS
            if next_token == tokenizer.special_tokens.get('<EOS>', -1):
                break

            generated.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)

            # Limit context
            if input_ids.size(1) >= 128:
                input_ids = input_ids[:, -128:]

    # Decode
    text = tokenizer.decode(generated)
    return text

def main():
    print("\n" + "="*70)
    print("TESTING QUALITY TRAINED MODEL")
    print("="*70)

    # Load model
    print("\nLoading model...")
    model, tokenizer, config = load_quality_model()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocabulary size: {len(tokenizer.token2id)}")
    print(f"Training tokens: word-level")

    # Test prompts
    prompts = [
        "once upon a time",
        "the cat",
        "in the forest",
        "there was a",
        "a little bird"
    ]

    print("\n" + "="*70)
    print("GENERATED TEXT SAMPLES")
    print("="*70)

    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")

        # Generate 3 samples with different temperatures
        for temp in [0.7, 1.0]:
            text = generate(model, tokenizer, prompt, max_length=30, temperature=temp)
            print(f"  [temp={temp}] {text}")

if __name__ == "__main__":
    main()
