import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import sys
from typing import List, Optional

# --- Model Configuration ---
vocab_size = 50281      # Or 100277
embed_size = 384         # Embedding dimension
hidden_size = 384        # Internal dimension for all layers
num_layers = 4           # Number of processing blocks
kernel_size = 3          # Size of the convolutional window

# --- Hierarchical Memory Configuration ---
num_memory_levels = 3    # L1 (working), L2 (episodic), L3 (long-term)
commit_interval = 64     # Commit from L1->L2 every 64 tokens. L2->L3 every 64*64=4096 tokens.

# --- Training Configuration ---
num_epochs = 20
learning_rate = 1e-3
batch_size = 4           # How many sequences to process in parallel
seq_length = 1024         # Context window length for training (can be much larger now)
predict_every_n_steps = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


# --- CUSTOM MODULES ---

class ExponentialTransform(nn.Module):
    """ The core custom activation function. """
    def __init__(self, features):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1) * 0.1 + 0.5)

    def forward(self, x):
        return x + self.alpha * x * torch.exp(-x**2 / 2)


class HierarchicalStateMemory(nn.Module):
    """
    Manages multiple levels of memory that update at different timescales, allowing for
    a massive context window with constant VRAM usage per token.
    """
    def __init__(self, embed_size, num_levels=3, commit_interval=64):
        super().__init__()
        self.embed_size = embed_size
        self.num_levels = num_levels
        self.commit_interval = commit_interval

        self.level_projections = nn.ModuleList([
            nn.Linear(embed_size, 3 * embed_size, bias=False) for _ in range(num_levels)
        ])
        self.output_projection = nn.Linear(num_levels * embed_size, embed_size, bias=False)

    def forward(self, x: torch.Tensor, past_states: Optional[List[torch.Tensor]] = None) -> (torch.Tensor, List[torch.Tensor]):
        x = x.transpose(1, 2)
        B, T, C = x.shape
        
        # Initialize states if not provided (start of training batch or generation)
        if past_states is None:
            states = [torch.zeros(B, C, device=x.device, dtype=x.dtype) for _ in range(self.num_levels)]
        else:
            states = past_states
        
        outputs: List[torch.Tensor] = []
        for t in range(T):
            x_t = x[:, t, :]
            
            # --- L1 Update (Working Memory) ---
            proj_l1 = self.level_projections[0](x_t)
            f_l1, i_l1, c_l1 = torch.chunk(proj_l1, 3, dim=-1)
            states[0] = torch.sigmoid(f_l1) * states[0] + torch.sigmoid(i_l1) * torch.tanh(c_l1)

            # --- Hierarchical Commit Mechanism ---
            for level in range(1, self.num_levels):
                current_commit_period = self.commit_interval ** level
                if (t + 1) % current_commit_period == 0:
                    state_from_below = states[level - 1].detach() # Stop gradients
                    proj = self.level_projections[level](state_from_below)
                    f, i, c = torch.chunk(proj, 3, dim=-1)
                    states[level] = torch.sigmoid(f) * states[level] + torch.sigmoid(i) * torch.tanh(c)
            
            # --- Read Mechanism ---
            combined_states = torch.cat(states, dim=-1)
            output_t = self.output_projection(combined_states)
            outputs.append(output_t)

        y = torch.stack(outputs, dim=1)
        return y.transpose(1, 2), states


class WaveGateHSM(nn.Module):
    """ The final architecture combining Convolutions and Hierarchical State Memory. """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'conv': nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=kernel_size - 1),
                'memory': HierarchicalStateMemory(hidden_size, num_memory_levels, commit_interval),
                'transform': ExponentialTransform(hidden_size),
                'norm1': nn.LayerNorm(hidden_size),
                'norm2': nn.LayerNorm(hidden_size)
            }))

        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets=None, past_layer_states: Optional[List[List[torch.Tensor]]] = None):
        B, T = idx.shape
        x = self.token_embedding(idx).transpose(1, 2)
        
        new_layer_states = []
        for i, layer in enumerate(self.layers):
            current_past_states = past_layer_states[i] if past_layer_states else None
            
            # Local processing (Convolution)
            residual = x
            x_norm = layer['norm1'](x.transpose(1, 2)).transpose(1, 2)
            x_conv = layer['conv'](x_norm)[:, :, :T]
            x = residual + x_conv
            
            # Global processing (Hierarchical Memory)
            residual = x
            x_norm = layer['norm2'](x.transpose(1, 2)).transpose(1, 2)
            x_mem, new_states = layer['memory'](x_norm, past_states=current_past_states)
            x = residual + x_mem
            new_layer_states.append(new_states)

            x = layer['transform'](x)

        logits = self.output_head(x.transpose(1, 2))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss, new_layer_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Efficient, stateful generation. At each step, we only process the newest token
        and pass the updated memory states to the next step. This is O(1) complexity.
        """
        self.eval()
        # Initialize memory states for all layers
        current_layer_states = None
        
        # First, process the initial context (if any) to warm up the states
        if idx.shape[1] > 1:
            _, _, current_layer_states = self(idx[:, :-1], past_layer_states=None)
            idx = idx[:, -1:] # The next token to be processed is the last one in the context

        for _ in range(max_new_tokens):
            # The forward pass now only processes one token at a time
            logits, _, current_layer_states = self(idx, past_layer_states=current_layer_states)
            
            # Get logits for the last token, sample, and append
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # The newly generated token becomes the input for the next step
            idx = idx_next
        
        # After generation is complete, the model's mode should be set back to train
        self.train()
        # The function now needs to return the generated indices, which are built up one by one.
        # This part of the logic needs to be handled by the caller.
        # For simplicity, we can collect them here.
        
        # Let's adjust generate to return the full sequence.
        # This requires a slight refactoring of the loop.
        
        # Corrected generate logic:
        self.eval()
        full_idx = idx
        past_states = None
        
        # Process prompt first to build initial state
        if idx.shape[1] > 1:
            prompt = idx[:, :-1]
            _, _, past_states = self(prompt, past_layer_states=None)
            idx = idx[:, -1:] # Start generation with the last token of the prompt

        for _ in range(max_new_tokens):
            logits, _, past_states = self(idx, past_layer_states=past_states)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.multinomial(probs, num_samples=1)
            full_idx = torch.cat((full_idx, idx), dim=1)

        self.train()
        return full_idx

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
    try:
        with open('input.txt', 'r', encoding='utf-8') as f: corpus = f.read()
    except FileNotFoundError:
        print("="*60 + "\nERROR: 'input.txt' not found.\nPlease create 'input.txt' and fill it with training text.\n" + "="*60)
        sys.exit()

    enc = tiktoken.get_encoding("p50k_base") #cl100k_base
    data = torch.tensor(enc.encode_ordinary(corpus), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - seq_length, (batch_size,))
        x = torch.stack([data[i:i+seq_length] for i in ix])
        y = torch.stack([data[i+1:i+seq_length+1] for i in ix])
        return x.to(device), y.to(device)

    model = WaveGateHSM()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("--- WaveGate Hierarchical State Memory (HSM) LLM Initialized ---")
    print(f"Using {num_memory_levels} memory levels with commit interval {commit_interval}.")
    print(f"Total Trainable Parameters: {total_params:,}")
    print("-" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    steps_per_epoch = len(train_data) // (batch_size * seq_length)
    total_iterations = 0

    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        for step in range(steps_per_epoch):
            xb, yb = get_batch('train')
            
            logits, loss, _ = model(xb, yb)
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