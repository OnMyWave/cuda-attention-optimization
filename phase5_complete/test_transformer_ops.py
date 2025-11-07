"""
Test correctness of LayerNorm and MLP CUDA implementations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

try:
    import transformer_ops
    CUDA_AVAILABLE = True
except ImportError:
    print("Error: transformer_ops module not found. Please build it first:")
    print("  python setup.py install")
    CUDA_AVAILABLE = False
    sys.exit(1)


def test_layer_norm(batch, seq_len, hidden_dim, rtol=1e-3, atol=1e-4):
    """Test LayerNorm implementation"""
    print(f"\nTesting LayerNorm: batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}")

    device = 'cuda'
    eps = 1e-5

    # Create inputs
    x = torch.randn(batch, seq_len, hidden_dim, device=device)
    gamma = torch.randn(hidden_dim, device=device)
    beta = torch.randn(hidden_dim, device=device)

    # PyTorch reference
    layer_norm_torch = nn.LayerNorm(hidden_dim, eps=eps, elementwise_affine=True, device=device)
    layer_norm_torch.weight.data = gamma.clone()
    layer_norm_torch.bias.data = beta.clone()

    with torch.no_grad():
        expected = layer_norm_torch(x)

    # CUDA implementation
    with torch.no_grad():
        actual = transformer_ops.layer_norm(x, gamma, beta, eps)

    # Check
    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    abs_diff = torch.abs(actual - expected)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()

    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")

    if is_close:
        print(f"  ✓ Test PASSED")
        return True
    else:
        print(f"  ✗ Test FAILED")
        print(f"  Sample values:")
        print(f"    Expected: {expected.flatten()[:5]}")
        print(f"    Actual:   {actual.flatten()[:5]}")
        return False


def test_mlp(batch, seq_len, hidden_dim, ff_dim, rtol=1e-3, atol=1e-4):
    """Test MLP implementation"""
    print(f"\nTesting MLP: batch={batch}, seq_len={seq_len}, hidden_dim={hidden_dim}, ff_dim={ff_dim}")

    device = 'cuda'

    # Create inputs
    x = torch.randn(batch, seq_len, hidden_dim, device=device)
    W1 = torch.randn(hidden_dim, ff_dim, device=device)
    b1 = torch.randn(ff_dim, device=device)
    W2 = torch.randn(ff_dim, hidden_dim, device=device)
    b2 = torch.randn(hidden_dim, device=device)

    # PyTorch reference
    with torch.no_grad():
        h = F.linear(x, W1.t(), b1)
        h = F.gelu(h)
        expected = F.linear(h, W2.t(), b2)

    # CUDA implementation
    with torch.no_grad():
        actual = transformer_ops.mlp_forward(x, W1, b1, W2, b2)

    # Check
    is_close = torch.allclose(actual, expected, rtol=rtol, atol=atol)

    abs_diff = torch.abs(actual - expected)
    max_abs_error = abs_diff.max().item()
    mean_abs_error = abs_diff.mean().item()

    print(f"  Max absolute error: {max_abs_error:.6e}")
    print(f"  Mean absolute error: {mean_abs_error:.6e}")

    if is_close:
        print(f"  ✓ Test PASSED")
        return True
    else:
        print(f"  ✗ Test FAILED")
        print(f"  Sample values:")
        print(f"    Expected: {expected.flatten()[:5]}")
        print(f"    Actual:   {actual.flatten()[:5]}")
        return False


def test_layer_norm_edge_cases():
    """Test LayerNorm edge cases"""
    print("\n" + "="*80)
    print("LayerNorm Edge Cases")
    print("="*80)

    tests = [
        (1, 1, 512),
        (1, 64, 512),
        (2, 128, 512),
        (4, 128, 768),
    ]

    all_passed = True
    for batch, seq_len, hidden_dim in tests:
        passed = test_layer_norm(batch, seq_len, hidden_dim)
        all_passed = all_passed and passed

    return all_passed


def test_mlp_configs():
    """Test MLP configurations"""
    print("\n" + "="*80)
    print("MLP Configurations")
    print("="*80)

    tests = [
        (1, 128, 512, 2048),
        (4, 128, 512, 2048),
        (2, 256, 512, 2048),
        (4, 128, 768, 3072),
    ]

    all_passed = True
    for batch, seq_len, hidden_dim, ff_dim in tests:
        passed = test_mlp(batch, seq_len, hidden_dim, ff_dim)
        all_passed = all_passed and passed

    return all_passed


def test_numerical_stability():
    """Test numerical stability"""
    print("\n" + "="*80)
    print("Numerical Stability Tests")
    print("="*80)

    batch, seq_len, hidden_dim = 2, 64, 512
    device = 'cuda'

    # Test LayerNorm with extreme values
    print("\nLayerNorm with large values:")
    x = torch.randn(batch, seq_len, hidden_dim, device=device) * 100
    gamma = torch.ones(hidden_dim, device=device)
    beta = torch.zeros(hidden_dim, device=device)

    with torch.no_grad():
        out = transformer_ops.layer_norm(x, gamma, beta)

    has_inf = torch.isinf(out).any().item()
    has_nan = torch.isnan(out).any().item()

    print(f"  Has inf: {has_inf}, Has nan: {has_nan}")
    test1_passed = not has_inf and not has_nan
    print(f"  {'✓ PASSED' if test1_passed else '✗ FAILED'}")

    # Test MLP with large values
    print("\nMLP with large values:")
    x = torch.randn(batch, seq_len, hidden_dim, device=device) * 10
    W1 = torch.randn(hidden_dim, 2048, device=device) * 0.1
    b1 = torch.zeros(2048, device=device)
    W2 = torch.randn(2048, hidden_dim, device=device) * 0.1
    b2 = torch.zeros(hidden_dim, device=device)

    with torch.no_grad():
        out = transformer_ops.mlp_forward(x, W1, b1, W2, b2)

    has_inf = torch.isinf(out).any().item()
    has_nan = torch.isnan(out).any().item()

    print(f"  Has inf: {has_inf}, Has nan: {has_nan}")
    test2_passed = not has_inf and not has_nan
    print(f"  {'✓ PASSED' if test2_passed else '✗ FAILED'}")

    return test1_passed and test2_passed


def main():
    """Run all tests"""
    print("="*80)
    print("Phase 5: Transformer Ops (LayerNorm, MLP) - Correctness Tests")
    print("="*80)

    if not CUDA_AVAILABLE:
        return

    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")

    results = []

    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)

    results.append(("LayerNorm Edge Cases", test_layer_norm_edge_cases()))
    results.append(("MLP Configurations", test_mlp_configs()))
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
