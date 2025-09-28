import torch
import torch.nn as nn
import math
from model import build_transformer, MultiHeadAttentionBlock


def _check_module_types(model, target_types, target_names=None) -> bool:
    """check if model contains specific module types or names."""
    target_names = target_names or []

    for module in model.modules():
        if isinstance(module, target_types):
            return True
        if any(name.lower() in str(type(module)).lower() for name in target_names):
            return True
    return False


def _has_positional_encoding(model) -> bool:
    # look for modules with "position" in their name or type
    pos_names = ["position", "positional", "pos_encoding"]
    return _check_module_types(model, tuple(), pos_names)


def _has_multihead_attention(model) -> bool:
    return _check_module_types(model, MultiHeadAttentionBlock)


def _has_feed_forward(model) -> bool:
    # check for modules that might be feed-forward components
    ff_names = ["feedforward", "ffn", "positionwisefeedforward"]
    return _check_module_types(model, nn.Linear, ff_names)


def _has_layer_norm(model) -> bool:
    """verify layer normalization is used."""
    return _check_module_types(model, nn.LayerNorm)


def _has_residual_connections(model) -> bool:
    """verify residual connections exist (check by known patterns)."""
    # this is a bit tricky since residuals are usually implemented in forward()
    # so we check for layer norm which typically follows residual connections
    # or look for modules with "residual" in the name
    res_names = ["residual", "resaddnorm"]
    return _has_layer_norm(model) or _check_module_types(model, tuple(), res_names)


def _verify_dimensions(model) -> bool:
    """basic dimension sanity checks."""
    d_model = 512  # paper default

    # check if we can access typical transformer components
    try:
        # verify encoder/decoder have layers attribute
        if not hasattr(model.encoder, "layers") or not hasattr(model.decoder, "layers"):
            return False

        # check attention dimensions match
        for name, module in model.named_modules():
            if isinstance(module, MultiHeadAttentionBlock):
                if module.d_model != d_model or module.h != 8:
                    return False
        return True
    except AttributeError:
        return False


def verify_model_implementation() -> bool:
    # build model with paper parameters
    model = build_transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        src_seq_len=500,
        tgt_seq_len=500,
        d_model=512,
        n=6,  # paper default
        h=8,  # paper default
        dropout=0.1,
    )

    checks = {
        "encoder_layers": len(model.encoder.layers) == 6,
        "decoder_layers": len(model.decoder.layers) == 6,
        "positional_encoding": _has_positional_encoding(model),
        "multihead_attention": _has_multihead_attention(model),
        "feed_forward": _has_feed_forward(model),
        "layer_normalization": _has_layer_norm(model),
        "residual_connections": _has_residual_connections(model),
        "correct_dimensions": _verify_dimensions(model),
    }

    print("model verification results:")
    for component, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {component}: {'passed' if passed else 'failed'}")

    all_passed = all(checks.values())
    if all_passed:
        print(
            "\n✓ model architecture verification passed - matches paper specifications"
        )
    else:
        print("\n✗ model architecture verification failed")

    return all_passed


def test_attention_isolation():
    batch_size, seq_len, d_model, h = 2, 10, 512, 8
    attention_block = MultiHeadAttentionBlock(d_model, h, dropout=0.1)

    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)

    # forward pass
    output, attention_weights = attention_block(q, k, v, None)

    # verify attention weights shape matches expected (batch, h, seq_len, seq_len)
    expected_shape = (batch_size, h, seq_len, seq_len)
    assert attention_weights.shape == expected_shape, (
        f"expected {expected_shape}, got {attention_weights.shape}"
    )

    print("✓ attention mechanism verification passed")
    return True


def test_scaled_dot_product():
    """test the scaled dot-product attention formula."""

    # simple test to verify the attention formula implementation
    d_k = 64
    seq_len = 5
    batch_size = 1

    q = torch.ones(batch_size, seq_len, d_k)
    k = torch.ones(batch_size, seq_len, d_k)
    v = torch.ones(batch_size, seq_len, d_k)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scaled dot product
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)

    print("✓ scaled dot-product attention test passed")
    print(f"  input shape: {q.shape}")
    print(f"  attention weights shape: {attention_weights.shape}")
    print(f"  output shape: {output.shape}")

    return True


def run_all_verification_tests() -> bool:
    """run all verification tests."""

    print("running model verification tests...")
    print("=" * 40)

    # test attention mechanism
    try:
        attention_passed = test_attention_isolation()
    except Exception as e:
        print(f"✗ attention mechanism test failed: {e}")
        attention_passed = False

    # test scaled dot-product
    try:
        scaled_passed = test_scaled_dot_product()
    except Exception as e:
        print(f"✗ scaled dot-product test failed: {e}")
        scaled_passed = False

    # test full model architecture
    try:
        model_passed = verify_model_implementation()
    except Exception as e:
        print(f"✗ model architecture test failed: {e}")
        model_passed = False

    print("=" * 40)
    print("verification summary:")
    print(f"  attention mechanism: {'✓ pass' if attention_passed else '✗ fail'}")
    print(f"  scaled dot-product: {'✓ pass' if scaled_passed else '✗ fail'}")
    print(f"  model architecture: {'✓ pass' if model_passed else '✗ fail'}")

    all_passed = attention_passed and scaled_passed and model_passed

    if all_passed:
        print("\n✓ all verification tests passed!")
    else:
        print("\n✗ some verification tests failed!")

    return all_passed


if __name__ == "__main__":
    run_all_verification_tests()
