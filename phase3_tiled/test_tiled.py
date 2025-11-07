"""
Test correctness of tiled CUDA attention implementation
"""

import torch
import torch.nn.functional as F
import math
import sys

try:
    import attention_tiled
    CUDA_AVAILABLE = True
except ImportError:
    print("Error: attention_tiled module not found. Please build it first:")
    print("  python setup.py install")
    CUDA_AVAILABLE = False
    sys.exit(1)


def pytorch_attention(Q, K, V):
    """Reference PyTorch implementation"""
    batch, seq_len, head_dim = Q.shape
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, V)
    return output


def test_tiled_attention(batch, seq_len, head_dim, tile_size=16, rtol=1e-3, atol=1e-4):
    """Test tiled attention implementation"""
    print(f"\nTesting: batch={batch}, seq_len={seq_len}, head_dim={head_dim}, tile_size={tile_size}")

    device = 'cuda'

    # Create random inputs
    Q = torch.randn(batch, seq_len, head_dim, device=device)
    K = torch.randn(batch, seq_len, head_dim, device=device)
    V = torch.randn(batch, seq_len, head_dim, device=device)

    # PyTorch reference
    with torch.no_grad():
        expected = pytorch_attention(Q, K, V)

    # Tiled CUDA implementation
    with torch.no_grad():
        actual = attention_tiled.forward(Q, K, V, tile_size)

    # Check shapes
    assert actual.shape == expected.shape, \
        f"Shape mismatch: {actual.shape} vs {expected.shape}"

    # Check values
    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    # Calculate error metrics
    abs_diff = torch.abs(actual - expected)
    rel_diff = abs_diff / (torch.abs(expected) + 1e-8)

    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()
    max_rel_error = rel_diff.max().item()
    mean_rel_error = rel_diff.mean().item()

    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")
    print(f"  Mean relative error: {mean_rel_error:.6e}")

    if is_close:
        print(f"  ✓ Test PASSED")
        return True
    else:
        print(f"  ✗ Test FAILED")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("Tiled CUDA Attention - Correctness Tests")
    print("="*80)

    if not CUDA_AVAILABLE:
        return

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Test configurations
    tests = [
        {'batch': 1, 'seq_len': 128, 'head_dim': 64, 'tile_size': 16},
        {'batch': 4, 'seq_len': 128, 'head_dim': 64, 'tile_size': 16},
        {'batch': 4, 'seq_len': 256, 'head_dim': 64, 'tile_size': 16},
        {'batch': 4, 'seq_len': 512, 'head_dim': 64, 'tile_size': 16},
        {'batch': 4, 'seq_len': 128, 'head_dim': 128, 'tile_size': 16},
    ]

    all_passed = True
    for config in tests:
        passed = test_tiled_attention(**config)
        all_passed = all_passed and passed

    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
    else:
        print("SOME TESTS FAILED! ✗")
    print("="*80)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
