import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import sys

# --- Model Configuration ---
vocab_size = 100277  # From tiktoken's cl100k_base
embed_size = 384     # Embedding dimension
hidden_size = 384    # Internal dimension for all layers
num_layers = 4       # Number of processing blocks
kernel_size = 3      # Size of the convolutional window

# --- Training Configuration ---
num_epochs = 20
learning_rate = 1e-3
batch_size = 4          # How many sequences to process in parallel
seq_length = 256        # Context window length
predict_every_n_steps = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# --- CUSTOM MODULES ---

class ExponentialTransform(nn.Module):
    """ The core custom activation function, now as a PyTorch Module. """
    def __init__(self, features):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1) * 0.1 + 0.5)

    def forward(self, x):
        return x + self.alpha * x * torch.exp(-x**2 / 2)

# --- NEW: Linear Complexity Attention Module ---
class StatefulGatedAttention(nn.Module):
    """
    An attention mechanism with O(N) linear complexity for training and O(1) constant 
    complexity for state updates during inference.

    This module processes a sequence token-by-token, updating a compressed hidden state
    that summarizes the entire past. Each token's output is a function of its own
    input and this summary state, allowing it to "attend" to the past globally.
    """
    def __init__(self, embed_size):
        super().__init__()
        # A single linear layer projects the input into multiple "gates" and a candidate value
        self.gate_projection = nn.Linear(embed_size, 3 * embed_size, bias=False)
        # An output projection to mix the state into the residual stream
        self.output_projection = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        # Input x is expected in (B, C, T) format from the CNN
        # We transpose to (B, T, C) for easier iteration over the time dimension
        x = x.transpose(1, 2)
        B, T, C = x.shape
        
        # Initialize the hidden state (the summary) for the sequence
        state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        
        outputs = []
        # This explicit loop is clear and demonstrates the linear complexity.
        # For performance on very long sequences, this could be optimized with torch.scan.
        for t in range(T):
            x_t = x[:, t, :] # Get the token at the current timestep (B, C)
            
            projected_gates = self.gate_projection(x_t)
            forget_gate, input_gate, candidate = torch.chunk(projected_gates, 3, dim=-1)
            
            # Apply activations to gates (sigmoid for 0-1 range) and candidate (tanh for -1 to 1)
            forget_gate = torch.sigmoid(forget_gate)
            input_gate = torch.sigmoid(input_gate)
            candidate = torch.tanh(candidate)
            
            # The core GRU-like state update: forget part of the old, add part of the new
            state = (forget_gate * state) + (input_gate * candidate)
            
            # The output for this timestep is a projection of the current state
            output_t = self.output_projection(state)
            outputs.append(output_t)

        y = torch.stack(outputs, dim=1) # Stack outputs to (B, T, C)
        return y.transpose(1, 2) # Return to (B, C, T) to match CNN convention


# --- MODIFIED: The Main Model Architecture ---
class WaveGateAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer = nn.Conv1d(
                in_channels=hidden_size, 
                out_channels=hidden_size, 
                kernel_size=kernel_size, 
                padding=kernel_size - 1 # Causal padding
            )
            attention_layer = StatefulGatedAttention(hidden_size)
            transform_layer = ExponentialTransform(hidden_size)
            
            # LayerNorms are crucial for stabilizing deep mixed-architecture models
            norm1 = nn.LayerNorm(hidden_size)
            norm2 = nn.LayerNorm(hidden_size)
            
            self.layers.append(nn.ModuleDict({
                'conv': conv_layer,
                'attention': attention_layer,
                'transform': transform_layer,
                'norm1': norm1,
                'norm2': norm2
            }))

        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) # (B, T, C)
        x = x.transpose(1, 2) # Switch to (B, C, T) for Conv1d
        
        for layer in self.layers:
            # --- Block 1: Local Processing via Causal Convolution ---
            residual = x
            # LayerNorm expects (B, ..., C), so we transpose
            x_norm = layer['norm1'](x.transpose(1, 2)).transpose(1, 2)
            x_conv = layer['conv'](x_norm)
            x_conv = x_conv[:, :, :T] # Enforce causality by slicing
            x = residual + x_conv
            
            # --- Block 2: Global Context via Stateful Gated Attention ---
            residual = x
            # LayerNorm expects (B, ..., C), so we transpose again
            x_norm = layer['norm2'](x.transpose(1, 2)).transpose(1, 2)
            x_attn = layer['attention'](x_norm)
            x = residual + x_attn

            # --- Final Non-linearity ---
            x = layer['transform'](x)

        x = x.transpose(1, 2) # Switch back to (B, T, C) for the output head
        logits = self.output_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train() # Set model back to training mode
        return idx

def safe_decode(enc, indices):
    byte_chunks = []
    for token in indices:
        try:
            byte_chunks.append(enc.decode_single_token_bytes(token))
        except KeyError:
            pass
    full_bytes = b"".join(byte_chunks)
    return full_bytes.decode('utf-8', errors='replace')

def train_llm():
    # --- Data Loading ---
    try:
        with open('input.txt', 'r', encoding='utf-8') as f: corpus = f.read()
    except FileNotFoundError:
        print("="*60 + "\nERROR: 'input.txt' not found.\nPlease create 'input.txt' and fill it with training text.\n" + "="*60)
        sys.exit()

    enc = tiktoken.get_encoding("cl100k_base")
    data = torch.tensor(enc.encode_ordinary(corpus), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - seq_length, (batch_size,))
        x = torch.stack([data[i:i+seq_length] for i in ix])
        y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
        return x.to(device), y.to(device)

    # --- Model Initialization ---
    model = WaveGateAttention() # <-- Using the new model class
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("--- WaveGate-Attention Hybrid LLM Initialized ---") # <-- Updated name
    print(f"Combines causal convolutions with linear-complexity stateful attention.")
    print(f"Training on {len(data):,} tokens from 'input.txt'.")
    print(f"Model Config: Embed={embed_size}, Hidden={hidden_size}, Layers={num_layers}")
    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 60)

    # --- Training Loop ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = len(train_data) // (batch_size * seq_length)
    total_iterations = 0

    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        for step in range(steps_per_epoch):
            xb, yb = get_batch('train')
            
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if total_iterations % 10 == 0:
                 print(f"Epoch {epoch+1}, Step {step}/{steps_per_epoch}: Loss {loss.item():.4f}")

            if total_iterations > 0 and total_iterations % predict_every_n_steps == 0:
                print("\n--- Generating Sample Text ---")
                start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
                generated_indices = model.generate(start_context, max_new_tokens=200)[0].tolist()
                generated_text = safe_decode(enc, generated_indices)
                print(generated_text)
                print("------------------------------\n")
            
            total_iterations += 1

    print("\n--- Training Complete ---")
    print("\n--- FINAL GENERATION ---")
    start_text = "The meaning of life is"
    start_context = torch.tensor(enc.encode(start_text), dtype=torch.long, device=device).unsqueeze(0)
    generated_indices = model.generate(start_context, max_new_tokens=500)[0].tolist()
    final_text = safe_decode(enc, generated_indices)
    print(final_text)

if __name__ == "__main__":
    train_llm()