"""
Phase 1: PyTorch Baseline Transformer Implementation
Small model: 4 layers, 512 hidden dim, 8 attention heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism"""

    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len, seq_len] or None
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)  # [batch, seq_len, hidden_dim]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Reshape for multi-head attention
        # [batch, seq_len, hidden_dim] -> [batch, seq_len, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Compute attention scores
        # [batch, num_heads, seq_len, head_dim] @ [batch, num_heads, head_dim, seq_len]
        # -> [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Apply softmax
        attn = F.softmax(scores, dim=-1)

        # Weighted sum with V
        # [batch, num_heads, seq_len, seq_len] @ [batch, num_heads, seq_len, head_dim]
        # -> [batch, num_heads, seq_len, head_dim]
        output = torch.matmul(attn, V)

        # Transpose back and reshape
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim]
        output = output.transpose(1, 2).contiguous()

        # Merge heads: [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, hidden_dim]
        output = output.view(batch_size, seq_len, self.hidden_dim)

        # Final linear projection
        output = self.out_proj(output)

        return output


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""

    def __init__(self, hidden_dim, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single Transformer block with attention and feed-forward"""

    def __init__(self, hidden_dim, num_heads, ff_dim=2048, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = FeedForward(hidden_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
            mask: [batch, seq_len, seq_len] or None
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class Transformer(nn.Module):
    """Complete Transformer model"""

    def __init__(self, vocab_size=10000, hidden_dim=512, num_layers=4,
                 num_heads=8, ff_dim=2048, max_seq_len=512, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len] - token indices
            mask: [batch, seq_len, seq_len] or None
        Returns:
            output: [batch, seq_len, vocab_size] - logits
        """
        batch_size, seq_len = x.shape

        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        # Embeddings
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization and output projection
        x = self.norm(x)
        logits = self.output(x)

        return logits


def create_causal_mask(seq_len, device='cuda'):
    """
    Create causal (autoregressive) mask
    Args:
        seq_len: sequence length
        device: device to create mask on
    Returns:
        mask: [1, 1, seq_len, seq_len] - lower triangular matrix
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
    return mask


if __name__ == "__main__":
    # Test the model
    print("Testing PyTorch Transformer...")

    # Model parameters
    batch_size = 4
    seq_len = 128
    vocab_size = 10000
    hidden_dim = 512
    num_layers = 4
    num_heads = 8

    # Check CUDA availability
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create model
    model = Transformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=2048,
        max_seq_len=512
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Create causal mask
    mask = create_causal_mask(seq_len, device)

    # Forward pass
    print(f"\nRunning forward pass...")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        output = model(input_ids, mask)

    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {vocab_size}]")

    # Test attention module separately
    print("\n" + "="*50)
    print("Testing MultiHeadAttention separately...")

    attn = MultiHeadAttention(hidden_dim, num_heads).to(device)
    x = torch.randn(batch_size, seq_len, hidden_dim, device=device)

    with torch.no_grad():
        attn_output = attn(x)

    print(f"Attention input shape: {x.shape}")
    print(f"Attention output shape: {attn_output.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {hidden_dim}]")

    print("\n✓ All tests passed!")
