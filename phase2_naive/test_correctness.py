"""
Test correctness of naive CUDA attention implementation
Compare against PyTorch reference
"""

import torch
import torch.nn.functional as F
import math
import sys

try:
    import attention_cuda
    CUDA_AVAILABLE = True
except ImportError:
    print("Warning: attention_cuda module not found. Please build it first:")
    print("  python setup.py install")
    CUDA_AVAILABLE = False
    sys.exit(1)


def pytorch_attention(Q, K, V):
    """
    Reference PyTorch implementation
    Args:
        Q, K, V: [batch, seq_len, head_dim]
    Returns:
        output: [batch, seq_len, head_dim]
    """
    batch, seq_len, head_dim = Q.shape

    # Q @ K^T / sqrt(d)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # Attention @ V
    output = torch.matmul(attn, V)

    return output


def test_attention(batch, seq_len, head_dim, rtol=1e-3, atol=1e-4):
    """
    Test attention implementation
    """
    print(f"\nTesting: batch={batch}, seq_len={seq_len}, head_dim={head_dim}")

    device = 'cuda'

    # Create random inputs
    Q = torch.randn(batch, seq_len, head_dim, device=device)
    K = torch.randn(batch, seq_len, head_dim, device=device)
    V = torch.randn(batch, seq_len, head_dim, device=device)

    # PyTorch reference
    with torch.no_grad():
        expected = pytorch_attention(Q, K, V)

    # CUDA implementation
    with torch.no_grad():
        actual = attention_cuda.forward(Q, K, V)

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

    print(f"  Shape: {actual.shape}")
    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")
    print(f"  Mean relative error: {mean_rel_error:.6e}")

    if is_close:
        print(f"  ✓ Test PASSED (rtol={rtol}, atol={atol})")
        return True
    else:
        print(f"  ✗ Test FAILED (rtol={rtol}, atol={atol})")

        # Print some sample values for debugging
        print("\n  Sample outputs (first 5 elements):")
        print(f"    Expected: {expected.flatten()[:5]}")
        print(f"    Actual:   {actual.flatten()[:5]}")
        print(f"    Diff:     {abs_diff.flatten()[:5]}")

        return False


def test_edge_cases():
    """Test edge cases"""
    print("\n" + "="*80)
    print("Testing Edge Cases")
    print("="*80)

    tests = [
        # (batch, seq_len, head_dim)
        (1, 1, 64),      # Minimal size
        (1, 2, 64),      # Small sequence
        (2, 8, 32),      # Small dimensions
        (1, 128, 128),   # Square attention matrix
    ]

    all_passed = True
    for batch, seq_len, head_dim in tests:
        passed = test_attention(batch, seq_len, head_dim)
        all_passed = all_passed and passed

    return all_passed


def test_standard_configs():
    """Test standard configurations"""
    print("\n" + "="*80)
    print("Testing Standard Configurations")
    print("="*80)

    tests = [
        # (batch, seq_len, head_dim)
        (1, 128, 64),    # Single sample
        (4, 128, 64),    # Small batch
        (8, 128, 64),    # Medium batch
        (4, 256, 64),    # Longer sequence
        (2, 512, 64),    # Very long sequence
        (4, 128, 128),   # Larger head dim
    ]

    all_passed = True
    for batch, seq_len, head_dim in tests:
        passed = test_attention(batch, seq_len, head_dim)
        all_passed = all_passed and passed

    return all_passed


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("\n" + "="*80)
    print("Testing Numerical Stability")
    print("="*80)

    batch, seq_len, head_dim = 2, 64, 64
    device = 'cuda'

    # Test 1: Large values
    print("\nTest 1: Large values (should not overflow)")
    Q = torch.randn(batch, seq_len, head_dim, device=device) * 10
    K = torch.randn(batch, seq_len, head_dim, device=device) * 10
    V = torch.randn(batch, seq_len, head_dim, device=device) * 10

    with torch.no_grad():
        expected = pytorch_attention(Q, K, V)
        actual = attention_cuda.forward(Q, K, V)

    has_inf = torch.isinf(actual).any().item()
    has_nan = torch.isnan(actual).any().item()

    print(f"  Has inf: {has_inf}, Has nan: {has_nan}")
    test1_passed = not has_inf and not has_nan and torch.allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"  {'✓ PASSED' if test1_passed else '✗ FAILED'}")

    # Test 2: Very small values
    print("\nTest 2: Small values")
    Q = torch.randn(batch, seq_len, head_dim, device=device) * 0.01
    K = torch.randn(batch, seq_len, head_dim, device=device) * 0.01
    V = torch.randn(batch, seq_len, head_dim, device=device) * 0.01

    with torch.no_grad():
        expected = pytorch_attention(Q, K, V)
        actual = attention_cuda.forward(Q, K, V)

    test2_passed = torch.allclose(actual, expected, rtol=1e-3, atol=1e-4)
    print(f"  {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    return test1_passed and test2_passed


def main():
    """Run all tests"""
    print("="*80)
    print("Naive CUDA Attention - Correctness Tests")
    print("="*80)

    if not CUDA_AVAILABLE:
        return

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")

    # Run tests
    results = []

    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)

    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Standard Configs", test_standard_configs()))
    results.append(("Numerical Stability", test_numerical_stability()))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    all_passed = all(passed for _, passed in results)

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
