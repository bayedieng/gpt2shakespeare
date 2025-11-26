import torch
from model import Gpt2Shakespeare
from tokenizers import Tokenizer
import torch.nn.functional as F

if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# Load trained weights
checkpoint = torch.load("./checkpoints/model_weights_50.pth", map_location=device)
model = Gpt2Shakespeare().to(device)
model.load_state_dict(checkpoint)
model.eval()

tokenizer: Tokenizer = Tokenizer.from_file("./tokenizer.json")
bos_id = tokenizer.token_to_id("<|bos|>")
eos_id = tokenizer.token_to_id("<|eos|>")

# Start with BOS token and keep feeding the full sequence so the model has context
generated_ids = [bos_id]
max_new_tokens = 128

with torch.no_grad():
    for _ in range(max_new_tokens):
        token_tensor = torch.tensor(generated_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = model(token_tensor)
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(probs, dim=-1).item()

        generated_ids.append(next_token_id)
        if next_token_id == eos_id:
            break

# Drop BOS for readability
decoded = tokenizer.decode(generated_ids[1:])
print(decoded)
    
