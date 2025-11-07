"""
Debug fused kernel - print intermediate values
"""

import torch
import torch.nn.functional as F
import math

try:
    import attention_fused
except ImportError:
    print("Error: attention_fused not found")
    exit(1)


def pytorch_attention_detailed(Q, K, V):
    """Reference with intermediate outputs"""
    batch, seq_len, head_dim = Q.shape

    # Q @ K^T
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    print(f"Scores shape: {scores.shape}")
    print(f"Scores[0,0,:5]: {scores[0,0,:5]}")
    print(f"Scores min/max: {scores.min():.4f} / {scores.max():.4f}")

    # Softmax
    attn = F.softmax(scores, dim=-1)
    print(f"Attn[0,0,:5]: {attn[0,0,:5]}")
    print(f"Attn[0,0,:].sum(): {attn[0,0,:].sum()}")

    # Attention @ V
    output = torch.matmul(attn, V)
    print(f"Output[0,0,:5]: {output[0,0,:5]}")

    return output


# Test with small size
batch, seq_len, head_dim = 1, 8, 64
device = 'cuda'  # Use default cuda device (cuda:0)

print("="*80)
print(f"Testing: batch={batch}, seq_len={seq_len}, head_dim={head_dim}")
print("="*80)

# Set seed for reproducibility
torch.manual_seed(42)

Q = torch.randn(batch, seq_len, head_dim, device=device)
K = torch.randn(batch, seq_len, head_dim, device=device)
V = torch.randn(batch, seq_len, head_dim, device=device)

print("\n" + "="*80)
print("PyTorch Reference:")
print("="*80)

with torch.no_grad():
    expected = pytorch_attention_detailed(Q, K, V)

print("\n" + "="*80)
print("CUDA Fused Kernel:")
print("="*80)

with torch.no_grad():
    actual = attention_fused.forward(Q, K, V)

print(f"Output[0,0,:5]: {actual[0,0,:5]}")

print("\n" + "="*80)
print("Comparison:")
print("="*80)

diff = torch.abs(expected - actual)
print(f"Max error: {diff.max():.6e}")
print(f"Mean error: {diff.mean():.6e}")
print(f"Expected[0,0,:5]: {expected[0,0,:5]}")
print(f"Actual[0,0,:5]:   {actual[0,0,:5]}")
print(f"Diff[0,0,:5]:     {diff[0,0,:5]}")

# Find where max error occurs
max_idx = torch.argmax(diff)
max_batch, max_seq, max_head = torch.unravel_index(max_idx, diff.shape)
print(f"\nMax error location: batch={max_batch}, seq={max_seq}, head={max_head}")
print(f"Expected at max: {expected[max_batch, max_seq, max_head]:.6f}")
print(f"Actual at max:   {actual[max_batch, max_seq, max_head]:.6f}")

is_close = torch.allclose(expected, actual, rtol=1e-3, atol=1e-4)
print(f"\nTest result: {'PASS' if is_close else 'FAIL'}")
