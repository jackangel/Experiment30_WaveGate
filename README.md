# WaveGate Language Model

WaveGate is a compact, efficient, and powerful autoregressive language model that explores an alternative to the dominant Transformer architecture. It is built upon a deep stack of **causal 1D convolutions** and features a novel, learnable gating mechanism called the **Exponential Transform**.

The name "WaveGate" is a nod to its two core components:
*   **Wave**: Inspired by WaveNet, which demonstrated the power of stacked causal convolutions for generating sequential data.
*   **Gate**: Refers to the custom `ExponentialTransform` activation function, which acts as a data-dependent, learnable gate to control information flow.

## Key Features

*   **Non-Transformer Architecture**: An entirely convolutional-based design, which can be highly parallelizable and efficient.
*   **Causal Convolutions**: Ensures that the prediction for a token at position `t` only depends on tokens from `1` to `t`, making it a true autoregressive model.
*   **Exponential Gating Unit**: A unique, custom activation function that non-linearly modulates the signal, allowing the network to selectively focus on or suppress features.
*   **Deep & Residual**: The architecture is a simple stack of blocks with residual connections, enabling stable training of deep networks.
*   **Simple and Understandable**: The codebase is small and self-contained, making it an excellent base for research and learning.

## Architecture Deep Dive

The WaveGate model processes a sequence of tokens through a series of identical blocks before predicting the next token.

```
Input Tokens
     |
     v
Token Embedding
     |
     v
[WaveGate Block 1]
     |
     v
[WaveGate Block 2]
     |
     v
     ...
     |
     v
[WaveGate Block N]
     |
     v
Output Linear Head
     |
     v
Logits (Predictions)
```

### The WaveGate Block

Each block in the stack is composed of three main operations: a causal convolution, the exponential gating unit, and a residual connection.



#### 1. Causal 1D Convolution

Instead of an attention mechanism, WaveGate uses a 1D convolution (`nn.Conv1d`) to aggregate information from neighboring tokens. To maintain autoregressive properties (i.e., prevent the model from "cheating" by looking at future tokens), we use a technique called **causal padding**.

The convolution is configured with `padding = kernel_size - 1`. This adds padding to the left of the sequence. After the convolution, the output is sliced to remove the extra steps from the right, ensuring the output sequence has the same length as the input and that each output step `t` only saw inputs up to `t`.

```python
# Causal Convolution Implementation
x = layer['conv'](x) # Applies convolution with left-padding
x = x[:, :, :T]      # Slices the output to enforce causality
```

This approach is highly efficient and can process all timesteps in parallel during training, unlike traditional RNNs.

#### 2. The Exponential Gating Unit (The "Special Sauce")

This is the most novel component of WaveGate. It's a learnable activation function applied after the convolution.

The transformation is defined by the formula:
`f(x) = x + α * x * exp(-x² / 2)`

Let's break it down:
*   `x`: The original input signal from the convolution (an identity path, similar to a residual connection).
*   `α`: A learnable scalar parameter (`alpha`) that scales the strength of the gating effect. The model learns the optimal value during training.
*   `exp(-x² / 2)`: This is a Gaussian function centered at zero. It has a value of `1` when `x` is `0` and rapidly decays as `x` moves away from `0`.

This term acts as a **soft, data-dependent gate**. It non-linearly modulates the input `x` based on its own magnitude. Features with small activations (close to zero) are transformed the most, while features with large positive or negative activations are left mostly untouched. This allows the network to learn complex patterns by selectively amplifying or dampening information flowing through it.

```python
class ExponentialTransform(nn.Module):
    def __init__(self, features):
        super().__init__()
        # Alpha is a learned parameter for the entire channel
        self.alpha = nn.Parameter(torch.randn(1) * 0.1 + 0.5)

    def forward(self, x):
        # The gating formula
        return x + self.alpha * x * torch.exp(-x**2 / 2)
```

#### 3. Residual Connection

Each WaveGate block is wrapped in a residual connection (`output = input + GatedConv(input)`). This is a standard and crucial technique for training deep neural networks. It allows gradients to flow directly through the network, preventing the vanishing gradient problem and enabling the model to learn deeper, more complex functions.

## How to Run

### Prerequisites

*   Python 3.8+
*   PyTorch (`torch`)
*   TikToken (`tiktoken`)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jackangel/Experiment30_WaveGate
    cd Experiment30_WaveGate
    ```

2.  **Install dependencies:**
    ```bash
    pip install torch tiktoken
    ```

3.  **Prepare the training data:**
    The script looks for a plain text file named `input.txt` in the same directory. Create this file and fill it with your training corpus (e.g., Shakespeare, Wikipedia articles, your own writing).
    ```bash
    # Example: Download "The Complete Works of William Shakespeare"
    wget https://www.gutenberg.org/files/100/100-0.txt -O input.txt
    ```

### Training

Simply run the Python script. The model will automatically use a GPU if `cuda` is available.

```bash
python wavegate.py
```

The script will periodically print the training loss and generate sample text to show the model's progress.

## Configuration

You can easily experiment with the model's architecture by modifying the configuration variables at the top of the script:

```python
# Model Configuration
vocab_size = 100277     # From tiktoken's cl100k_base
embed_size = 384        # Embedding dimension
hidden_size = 384       # Internal dimension of the CNN layers
num_layers = 4          # Number of WaveGate blocks
kernel_size = 3         # Size of the convolutional window

# Training Configuration
num_epochs = 20
learning_rate = 1e-3
batch_size = 4
seq_length = 256
```
