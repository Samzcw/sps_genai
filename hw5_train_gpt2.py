from helper_lib.gpt2_squad_trainer import train_gpt2_squad

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

hw5_model, hw5_tokenizer = train_gpt2_squad(checkpoint_dir="hw5_checkpoints/gpt2-squad-finetuned")

print("Training complete. Model and tokenizer saved to hw5_checkpoints/gpt2-squad-finetuned")