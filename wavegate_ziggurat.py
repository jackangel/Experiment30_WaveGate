import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
import time
import sys
import math

# ------------------- Model Configuration -------------------
vocab_size = 50281
embed_size = 384
hidden_size = 384
num_layers = 6
kernel_size = 3

# --- Ziggurat-Memory Specific Configuration ---
bucket_size = 256
max_context_length = bucket_size # Note: The effective context is now larger due to memory
num_attention_heads = 8
num_memory_tokens = 32      # Number of "Abstract Concept" buckets
num_verbatim_tokens = 16   # Number of "most important raw tokens" to keep
fuzzy_loss_weight = 0.5

# ------------------- Training Configuration -------------------
num_epochs = 100
learning_rate = 3e-4
weight_decay = 0.1
batch_size = 4
predict_every_n_steps = 200
grad_clip_norm = 1.0

enc = tiktoken.get_encoding("p50k_base")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ------------------- Custom Activation Function -------------------
class ExponentialTransform(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(1) * 0.1 + 0.5)
    def forward(self, x):
        return x + self.alpha * x * torch.exp(-x**2 / 2)

# ------------------- Ziggurat-Memory Model Architecture -------------------
class HybridCNN_LLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        # Positional embedding is only for the main bucket tokens
        self.pos_embedding = nn.Embedding(max_context_length, embed_size)

        self.local_attention_block = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_attention_heads,
            dim_feedforward=4 * embed_size, dropout=0.1,
            activation=F.gelu, batch_first=True, norm_first=True
        )
        self.initial_projection = nn.Conv1d(in_channels=embed_size, out_channels=hidden_size, kernel_size=1)

        # --- ZIGGURAT-MEMORY CORE ---
        self.summary_query = nn.Parameter(torch.randn(1, num_memory_tokens, embed_size))
        self.summary_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_attention_heads, batch_first=True)
        self.memory_retrieval_attention = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_attention_heads, batch_first=True)
        self.retrieval_query_generator = nn.Linear(embed_size * num_memory_tokens, num_verbatim_tokens * embed_size)
        # --- END ZIGGURAT-MEMORY CORE ---

        self.context_combiner = nn.Linear(embed_size * num_memory_tokens, embed_size)
        self.film_generator = nn.Sequential(
            nn.Linear(embed_size * num_memory_tokens, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 2 * hidden_size * num_layers)
        )
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_layer_1 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=1)
            conv_layer_2 = nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=kernel_size, padding=0)
            transform_layer = ExponentialTransform(hidden_size)
            norm_layer = nn.LayerNorm(hidden_size)
            self.layers.append(nn.ModuleDict({
                'norm': norm_layer, 'conv1': conv_layer_1,
                'conv2': conv_layer_2, 'transform': transform_layer
            }))
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

        # Fuzzy loss components are preserved
        self.fuzzy_projector = nn.Linear(hidden_size, hidden_size)
        self.fuzzy_head = nn.Linear(hidden_size, vocab_size)
        self.fuzzy_mem_combiner = nn.Linear(embed_size * num_memory_tokens, hidden_size)

    def _create_multi_vector_summary(self, embeddings_with_pos):
        B = embeddings_with_pos.shape[0]
        query = self.summary_query.expand(B, -1, -1)
        summary_vectors, _ = self.summary_attention(query=query, key=embeddings_with_pos, value=embeddings_with_pos)
        return summary_vectors

    def forward(self, main_idx, past_summary=None, past_verbatim_memory=None, past_bucket_embeddings=None, targets=None, pos_offset=0):
        B, T = main_idx.shape
        pos_indices = torch.arange(pos_offset, pos_offset + T, device=device).unsqueeze(0) % max_context_length
        
        # Initialize states if they are not provided (start of an epoch)
        if past_summary is None: 
            past_summary = torch.zeros(B, num_memory_tokens, embed_size, device=main_idx.device)
        if past_verbatim_memory is None:
            past_verbatim_memory = torch.zeros(B, num_verbatim_tokens, embed_size, device=main_idx.device)

        flat_context_summary = past_summary.reshape(B, -1)
        
        main_embed_with_pos = self.token_embedding(main_idx) + self.pos_embedding(pos_indices)
        
        # --- NEW: Select new verbatim memory from the *previous* bucket's embeddings ---
        # If there are past embeddings, we select from them; otherwise, we carry the old memory forward.
        if past_bucket_embeddings is not None:
            retrieval_queries = self.retrieval_query_generator(flat_context_summary).reshape(B, num_verbatim_tokens, embed_size)
            current_verbatim_memory, _ = self.memory_retrieval_attention(query=retrieval_queries, key=past_bucket_embeddings, value=past_bucket_embeddings)
        else:
            current_verbatim_memory = past_verbatim_memory

        # --- MODIFIED: Integrate Verbatim Memory into the main processing stream ---
        context_signal = self.context_combiner(flat_context_summary)
        x_conditioned = main_embed_with_pos + context_signal.unsqueeze(1)
        
        full_input_sequence = torch.cat([current_verbatim_memory, x_conditioned], dim=1)
        
        # Create a combined attention mask for the TransformerEncoderLayer
        S = T + num_verbatim_tokens
        full_attn_mask = torch.full((S, S), float("-inf"), device=device)
        # Allow memory to see everything, and main tokens to see all memory
        full_attn_mask[:, :num_verbatim_tokens] = 0.
        # Create causal mask for the main tokens among themselves
        causal_part = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        full_attn_mask[num_verbatim_tokens:, num_verbatim_tokens:] = causal_part

        x_attn = self.local_attention_block(full_input_sequence, src_mask=full_attn_mask)
        x_attn_main = x_attn[:, num_verbatim_tokens:, :] # Slice out only the main token outputs
        
        # Create the abstract summary for the *current* bucket, to be used in the *next* step.
        current_summary = self._create_multi_vector_summary(main_embed_with_pos)
        
        # --- The rest of the network proceeds as before, using x_attn_main ---
        film_params = self.film_generator(flat_context_summary)
        gammas = film_params[:, :hidden_size * num_layers].reshape(B, num_layers, hidden_size, 1)
        betas = film_params[:, hidden_size * num_layers:].reshape(B, num_layers, hidden_size, 1)
        
        x = self.initial_projection(x_attn_main.transpose(1, 2))
        
        for i, layer in enumerate(self.layers):
            residual = x
            x_norm = layer['norm'](x.transpose(1, 2)).transpose(1, 2)
            x = layer['conv1'](x_norm)
            x_padded = F.pad(x, (kernel_size - 1, 0))
            x = layer['conv2'](x_padded)
            gamma_i = gammas[:, i, :, :]
            beta_i = betas[:, i, :, :]
            x = x * gamma_i + beta_i
            x = layer['transform'](x)
            x = x + residual
        
        x = self.final_norm(x.transpose(1, 2))
        main_logits = self.output_head(x)
        
        loss = None
        if targets is not None:
            main_loss = F.cross_entropy(main_logits.reshape(-1, main_logits.size(-1)), targets.reshape(-1), ignore_index=-1)
            flat_main_summary = current_summary.reshape(B, -1)
            summary_of_main_bucket_for_loss = self.fuzzy_mem_combiner(flat_main_summary)
            projected_main_summary = self.fuzzy_projector(summary_of_main_bucket_for_loss)
            fuzzy_logits_main = self.fuzzy_head(projected_main_summary.unsqueeze(1).expand(-1, T, -1))
            fuzzy_loss_main = F.cross_entropy(fuzzy_logits_main.reshape(-1, fuzzy_logits_main.size(-1)), targets.reshape(-1), ignore_index=-1)
            loss = main_loss + (fuzzy_loss_weight * fuzzy_loss_main)

        return main_logits, loss, current_summary, current_verbatim_memory, main_embed_with_pos

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # Generation needs to manage all states now
        self.eval()
        past_summary = None
        past_verbatim_memory = None
        
        # Process the initial prompt to build the starting states
        if idx.shape[1] > 0:
            prompt_embeddings = None
            for i in range(0, idx.shape[1], bucket_size):
                chunk = idx[:, i:i+bucket_size]
                if chunk.shape[1] == 0: continue
                _, _, past_summary, past_verbatim_memory, prompt_embeddings = self(
                    chunk, past_summary=past_summary, past_verbatim_memory=past_verbatim_memory, past_bucket_embeddings=prompt_embeddings)

        generated_idx = idx
        cnn_caches = [torch.zeros(1, hidden_size, kernel_size - 1, device=device) for _ in self.layers]

        for _ in range(max_new_tokens):
            idx_cond = generated_idx[:, -1:]
            current_pos = generated_idx.shape[1] - 1
            
            # --- Single-token forward pass logic, adapted for Ziggurat-Memory ---
            pos_indices = torch.tensor([[current_pos]], device=device) % max_context_length
            embeds = self.token_embedding(idx_cond) + self.pos_embedding(pos_indices)

            if past_summary is None: past_summary = torch.zeros(1, num_memory_tokens, embed_size, device=device)
            if past_verbatim_memory is None: past_verbatim_memory = torch.zeros(1, num_verbatim_tokens, embed_size, device=device)
            
            flat_context_summary = past_summary.reshape(1, -1)
            context_signal = self.context_combiner(flat_context_summary)
            conditioned_embeds = embeds + context_signal.unsqueeze(1)
            
            full_input_sequence = torch.cat([past_verbatim_memory, conditioned_embeds], dim=1)
            x_attn = self.local_attention_block(full_input_sequence)
            next_token_features = x_attn[:, -1:, :] # Get output for the last token

            film_params = self.film_generator(flat_context_summary)
            gammas = film_params[:, :hidden_size * num_layers].reshape(1, num_layers, hidden_size, 1)
            betas = film_params[:, hidden_size * num_layers:].reshape(1, num_layers, hidden_size, 1)
            
            x = self.initial_projection(next_token_features.transpose(1, 2))
            
            for i, layer in enumerate(self.layers):
                residual = x
                x_norm = layer['norm'](x.transpose(1, 2)).transpose(1, 2)
                x = layer['conv1'](x_norm)
                x_with_context = torch.cat([cnn_caches[i], x], dim=2)
                cnn_caches[i] = x_with_context[:, :, 1:]
                x = layer['conv2'](x_with_context)
                x = x * gammas[:, i, :, :] + betas[:, i, :, :]
                x = layer['transform'](x)
                x = x + residual
            
            logits = self.output_head(self.final_norm(x.transpose(1, 2)))
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_idx = torch.cat((generated_idx, next_token), dim=1)

            # Periodically update the summary states for very long generations
            if generated_idx.shape[1] % bucket_size == 0:
                 chunk = generated_idx[:, -bucket_size:]
                 _, _, past_summary, past_verbatim_memory, _ = self(chunk, past_summary=past_summary, past_verbatim_memory=past_verbatim_memory, past_bucket_embeddings=embeds)

        self.train()
        return generated_idx

def safe_decode(enc, indices):
    byte_chunks = []
    for token in indices:
        try: byte_chunks.append(enc.decode_single_token_bytes(token))
        except KeyError: pass
    full_bytes = b"".join(byte_chunks)
    return full_bytes.decode('utf-8', errors='replace')

def train_llm():
    # ... data loading ...
    try:
        with open('input.txt', 'r', encoding='utf-8') as f: corpus = f.read()
    except FileNotFoundError:
        print("="*60 + "\nERROR: 'input.txt' not found.\n" + "="*60); sys.exit()
    data = torch.tensor(enc.encode_ordinary(corpus), dtype=torch.long)
    n = int(0.9 * len(data)); train_data, val_data = data[:n], data[n:]

    def get_batch(split, start_index):
        data_source = train_data if split == 'train' else val_data
        end_index = start_index + batch_size * bucket_size
        if end_index + 1 > len(data_source): return None, None
        x_list = [data_source[start_index + i*bucket_size : start_index + i*bucket_size + bucket_size] for i in range(batch_size)]
        y_list = [data_source[start_index + i*bucket_size + 1 : start_index + i*bucket_size + bucket_size + 1] for i in range(batch_size)]
        return torch.stack(x_list).to(device), torch.stack(y_list).to(device)

    model = HybridCNN_LLM()
    model.to(device)
    print(f"--- Ziggurat-Memory Architecture Initialized (v13 - Verbatim Memory) ---")
    print(f"Total Trainable Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("-" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = (len(train_data) // (batch_size * bucket_size)) - 1
    total_training_steps = num_epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_training_steps)
    
    total_iterations = 0
    for epoch in range(num_epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")
        model.train()
        
        # --- MODIFIED: Initialize all persistent states for the epoch ---
        past_summary, past_verbatim_memory, past_bucket_embeddings = None, None, None
        
        for step in range(steps_per_epoch):
            current_pos = step * bucket_size
            main_bx, main_by = get_batch('train', current_pos)
            if main_bx is None: break

            logits, loss, current_summary, current_verbatim, current_embeds = model(
                main_idx=main_bx, 
                past_summary=past_summary, 
                past_verbatim_memory=past_verbatim_memory,
                past_bucket_embeddings=past_bucket_embeddings,
                targets=main_by
            )
            
            # --- MODIFIED: Update all states for the next iteration, detaching them all ---
            past_summary = current_summary.detach()
            past_verbatim_memory = current_verbatim.detach()
            past_bucket_embeddings = current_embeds.detach()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            scheduler.step()

            if total_iterations % 10 == 0:
                 print(f"Epoch {epoch+1}, Step {step+1}/{steps_per_epoch}: Loss {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            
            if total_iterations > 0 and total_iterations % predict_every_n_steps == 0:
                print("\n--- Generating Sample Text ---")
                start_context = torch.tensor(enc.encode("Dr. Alistair Finch reviewed his notes; the subject, a curious artifact from the dig site, defied all logic."), dtype=torch.long, device=device).unsqueeze(0)
                generated_indices = model.generate(start_context, max_new_tokens=128)[0].tolist()
                print(safe_decode(enc, generated_indices))
                print("------------------------------\n")
            
            total_iterations += 1
        if total_iterations >= total_training_steps: break
    print("\n--- Training Complete ---")

if __name__ == "__main__":
    train_llm()