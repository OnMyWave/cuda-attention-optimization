"""
Phase 6: Integrated Optimized Transformer
Combines all CUDA-optimized components into a full Transformer block
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directories to path to import CUDA extensions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase4_optimized'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'phase5_complete'))

try:
    import attention_fused
    ATTENTION_AVAILABLE = True
except ImportError:
    print("Warning: attention_fused not available. Using PyTorch attention.")
    ATTENTION_AVAILABLE = False

try:
    import transformer_ops
    TRANSFORMER_OPS_AVAILABLE = True
except ImportError:
    print("Warning: transformer_ops not available. Using PyTorch ops.")
    TRANSFORMER_OPS_AVAILABLE = False


class OptimizedMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention using optimized CUDA kernels
    """

    def __init__(self, hidden_dim, num_heads, use_cuda=True):
        super().__init__()
        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_cuda = use_cuda and ATTENTION_AVAILABLE

        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        if self.use_cuda and self.num_heads == 1:
            # Use fused CUDA kernel (single head)
            output = attention_fused.forward(Q, K, V)
        else:
            # Multi-head: reshape and use PyTorch or CUDA per-head
            # [batch, seq_len, hidden_dim] -> [batch, seq_len, num_heads, head_dim]
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

            # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            V = V.transpose(1, 2)

            # Process each head
            outputs = []
            for h in range(self.num_heads):
                Q_h = Q[:, h, :, :].contiguous()
                K_h = K[:, h, :, :].contiguous()
                V_h = V[:, h, :, :].contiguous()

                if self.use_cuda:
                    out_h = attention_fused.forward(Q_h, K_h, V_h)
                else:
                    # Fallback to PyTorch
                    import math
                    scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
                    attn = torch.softmax(scores, dim=-1)
                    out_h = torch.matmul(attn, V_h)

                outputs.append(out_h)

            # Stack and reshape
            output = torch.stack(outputs, dim=1)  # [batch, num_heads, seq_len, head_dim]
            output = output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
            output = output.view(batch_size, seq_len, self.hidden_dim)

        # Final projection
        output = self.out_proj(output)
        return output


class OptimizedLayerNorm(nn.Module):
    """LayerNorm using optimized CUDA kernel"""

    def __init__(self, hidden_dim, eps=1e-5, use_cuda=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.use_cuda = use_cuda and TRANSFORMER_OPS_AVAILABLE

        self.gamma = nn.Parameter(torch.ones(hidden_dim))
        self.beta = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        if self.use_cuda:
            return transformer_ops.layer_norm(x, self.gamma, self.beta, self.eps)
        else:
            # Fallback to PyTorch
            return nn.functional.layer_norm(x, (self.hidden_dim,), self.gamma, self.beta, self.eps)


class OptimizedMLP(nn.Module):
    """MLP using optimized CUDA kernel"""

    def __init__(self, hidden_dim, ff_dim=2048, use_cuda=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.use_cuda = use_cuda and TRANSFORMER_OPS_AVAILABLE

        self.W1 = nn.Parameter(torch.randn(hidden_dim, ff_dim) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(ff_dim))
        self.W2 = nn.Parameter(torch.randn(ff_dim, hidden_dim) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x):
        if self.use_cuda:
            return transformer_ops.mlp_forward(x, self.W1, self.b1, self.W2, self.b2)
        else:
            # Fallback to PyTorch
            import torch.nn.functional as F
            h = F.linear(x, self.W1.t(), self.b1)
            h = F.gelu(h)
            return F.linear(h, self.W2.t(), self.b2)


class OptimizedTransformerBlock(nn.Module):
    """
    Complete Transformer block with CUDA-optimized components
    """

    def __init__(self, hidden_dim, num_heads, ff_dim=2048, dropout=0.1, use_cuda=True):
        super().__init__()

        self.attention = OptimizedMultiHeadAttention(hidden_dim, num_heads, use_cuda)
        self.norm1 = OptimizedLayerNorm(hidden_dim, use_cuda=use_cuda)
        self.mlp = OptimizedMLP(hidden_dim, ff_dim, use_cuda)
        self.norm2 = OptimizedLayerNorm(hidden_dim, use_cuda=use_cuda)
        self.dropout = nn.Dropout(dropout)

        self.use_cuda = use_cuda

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        # Self-attention with residual and norm
        attn_out = self.attention(x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP with residual and norm
        mlp_out = self.mlp(x)
        x = x + self.dropout(mlp_out)
        x = self.norm2(x)

        return x


class OptimizedTransformer(nn.Module):
    """
    Complete Transformer model with CUDA optimization
    """

    def __init__(self, vocab_size=10000, hidden_dim=512, num_layers=4,
                 num_heads=8, ff_dim=2048, max_seq_len=512, dropout=0.1,
                 use_cuda=True):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.use_cuda = use_cuda

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            OptimizedTransformerBlock(hidden_dim, num_heads, ff_dim, dropout, use_cuda)
            for _ in range(num_layers)
        ])

        # Output
        self.norm = OptimizedLayerNorm(hidden_dim, use_cuda=use_cuda)
        self.output = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len] - token indices
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        token_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.norm(x)
        logits = self.output(x)

        return logits

    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


def test_optimized_transformer():
    """Test the optimized transformer"""
    print("="*80)
    print("Testing Optimized Transformer")
    print("="*80)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = 'cuda'
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Attention CUDA available: {ATTENTION_AVAILABLE}")
    print(f"Transformer Ops CUDA available: {TRANSFORMER_OPS_AVAILABLE}")

    # Model parameters
    batch_size = 4
    seq_len = 128
    vocab_size = 10000
    hidden_dim = 512
    num_layers = 4
    num_heads = 8

    # Create model
    print(f"\nCreating model...")
    model = OptimizedTransformer(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=2048,
        max_seq_len=512,
        use_cuda=True
    ).to(device)

    print(f"Number of parameters: {model.get_num_params():,}")

    # Create input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Forward pass
    print(f"\nRunning forward pass...")
    print(f"Input shape: {input_ids.shape}")

    with torch.no_grad():
        output = model(input_ids)

    print(f"Output shape: {output.shape}")
    print(f"Expected: [{batch_size}, {seq_len}, {vocab_size}]")

    # Check for NaN or Inf
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()

    print(f"\nHas NaN: {has_nan}")
    print(f"Has Inf: {has_inf}")

    if not has_nan and not has_inf:
        print("\n✓ Test PASSED!")
        return True
    else:
        print("\n✗ Test FAILED!")
        return False


if __name__ == "__main__":
    success = test_optimized_transformer()
    sys.exit(0 if success else 1)
