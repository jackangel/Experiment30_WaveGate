import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import sys

# Model Configuration
vocab_size = 50281  #100277
embed_size = 384
hidden_size = 384
num_layers = 6
kernel_size = 3
pool_factors = [4, 16]
clear_view_size = 128 

# Training Configuration
num_epochs = 40
learning_rate = 1e-4
batch_size = 4
seq_length = 256
predict_every_n_steps = 200

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class ExponentialTransform(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1) * 0.1 + 0.5)

    def forward(self, x):
        return x + self.alpha * x * torch.exp(-x**2 / 2)

class HybridCNN_LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            combined_feature_size = hidden_size * (1 + len(pool_factors))
            feature_combiner = nn.Conv1d(in_channels=combined_feature_size, out_channels=hidden_size, kernel_size=1)
            conv_layer = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=kernel_size - 1)
            transform_layer = ExponentialTransform(hidden_size)
            
            self.layers.append(nn.ModuleDict({
                'combiner': feature_combiner, 'conv': conv_layer, 'transform': transform_layer
            }))
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, idx, targets=None, past_states=None):
        B, T = idx.shape
        x = self.token_embedding(idx).transpose(1, 2)

        x_for_pooling = x.detach()

        is_generating = past_states is not None
        new_states = [] if is_generating else None

        for i, layer in enumerate(self.layers):
            residual = x
            features_to_combine = [x]
            
            layer_past_states = past_states[i] if is_generating else None
            layer_new_states = {} if is_generating else None

            for factor in pool_factors:
                upsampled_context = torch.zeros_like(x)
                
                if is_generating:
                    empty_past = torch.zeros(B, hidden_size, 0, device=x.device)
                    recent_clear_x = layer_past_states.get('recent_clear_x', empty_past)
                    past_summary = layer_past_states.get(f'{factor}_summary', empty_past)
                    unpooled_x = layer_past_states.get(f'{factor}_unpooled', empty_past)
                    
                    updated_clear_x = torch.cat([recent_clear_x, x_for_pooling], dim=2)
                    
                    x_to_be_pooled = empty_past
                    if updated_clear_x.shape[2] > clear_view_size:
                        overflow = updated_clear_x.shape[2] - clear_view_size
                        x_to_be_pooled = updated_clear_x[:, :, :overflow]
                        final_clear_x = updated_clear_x[:, :, overflow:]
                    else:
                        final_clear_x = updated_clear_x
                    
                    layer_new_states['recent_clear_x'] = final_clear_x
                    combined_for_pooling = torch.cat([unpooled_x, x_to_be_pooled], dim=2)
                    num_new_blocks = combined_for_pooling.shape[2] // factor
                    
                    new_full_summary = past_summary
                    if num_new_blocks > 0:
                        x_to_pool = combined_for_pooling[:, :, :num_new_blocks * factor]
                        new_summary_part = F.avg_pool1d(x_to_pool, kernel_size=factor, stride=factor)
                        new_full_summary = torch.cat([past_summary, new_summary_part], dim=2)
                    
                    new_unpooled = combined_for_pooling[:, :, num_new_blocks * factor:]
                    layer_new_states[f'{factor}_summary'] = new_full_summary
                    layer_new_states[f'{factor}_unpooled'] = new_unpooled
                    
                    if new_full_summary.shape[2] > 0:
                        upsampled_context = new_full_summary[:, :, -1:].expand(-1, -1, T)

                else: # Training Path
                    if T > clear_view_size:
                        fuzzy_part = x_for_pooling[:, :, :-clear_view_size]
                        pooled = F.avg_pool1d(fuzzy_part, kernel_size=factor, stride=factor)
                        if pooled.shape[2] > 0:
                            fuzzy_len = fuzzy_part.shape[2]
                            upsampled_fuzzy = F.interpolate(pooled, size=fuzzy_len, mode='nearest')
                            clear_context = pooled[:, :, -1:].expand(-1, -1, clear_view_size)
                            upsampled_context = torch.cat([upsampled_fuzzy, clear_context], dim=2)
                
                features_to_combine.append(upsampled_context)
            
            if is_generating:
                new_states.append(layer_new_states)

            combined_features = torch.cat(features_to_combine, dim=1)
            x = layer['combiner'](combined_features)
            x = layer['conv'](x)[:, :, :T]
            x = layer['transform'](x)
            x = x + residual # The main signal path still evolves with residual connections
            
        logits = self.output_head(x.transpose(1, 2))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        past_states = [{} for _ in self.layers]
        
        if idx.shape[1] > 1:
            _, _, past_states = self(idx[:, :-1], past_states=past_states)
            current_idx = idx[:, -1:]
        else:
            current_idx = idx

        for _ in range(max_new_tokens):
            logits, _, past_states = self(current_idx, past_states=past_states)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            current_idx = idx_next

        self.train()
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

    model = HybridCNN_LLM()
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("--- Deep Causal CNN LLM with Tiered Memory ---")
    print(f"Training on {len(data):,} tokens from 'input.txt'.")
    # Updated print statement
    print(f"Model Config: Embed={embed_size}, Hidden={hidden_size}, Layers={num_layers}, ClearView={clear_view_size}, PoolFactors={pool_factors}")
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
            
            # Note: The forward pass now returns three items
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