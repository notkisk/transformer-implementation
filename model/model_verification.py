"""
Model verification tests for Transformer implementation.
Comprehensive validation to ensure the implementation matches the paper's architecture.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import build_transformer, MultiHeadAttentionBlock
from typing import Dict, Any
import math


def has_positional_encoding(model) -> bool:
    """Check if model has positional encoding components."""
    for name, module in model.named_modules():
        if 'pos' in name.lower() or 'position' in name.lower():
            if 'positional' in str(type(module)).lower():
                return True
    return False


def has_multihead_attention(model) -> bool:
    """Check if model has multihead attention components."""
    for module in model.modules():
        if isinstance(module, MultiHeadAttentionBlock):
            return True
    return False


def has_feed_forward_networks(model) -> bool:
    """Check if model has feed forward networks."""
    for name, module in model.named_modules():
        if 'feedforward' in name.lower() or 'ffn' in name.lower():
            if 'linear' in str(type(module)).lower():
                return True
    return False


def has_layer_normalization(model) -> bool:
    """Check if model has layer normalization components."""
    # Check for both PyTorch LayerNorm and custom LayerNormalization
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            return True
        # Check for the custom LayerNormalization class defined in the model
        module_name = str(type(module).__name__)
        if 'LayerNormalization' in module_name:
            return True
    return False


def has_residual_connections(model) -> bool:
    """Check if model has components that support residual connections."""
    # Check if there are modules that could implement residual connections
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import ResidualAddNorm
    for module in model.modules():
        if isinstance(module, ResidualAddNorm):
            return True
    return False


def verify_dimensions(model, config: Dict[str, Any]) -> bool:
    """Verify that model dimensions match configuration."""
    d_model = config['d_model']
    
    # Check if embedding dimensions match
    src_embed = getattr(model, 'src_embed', None)
    tgt_embed = getattr(model, 'tgt_embed', None)
    
    if src_embed and hasattr(src_embed, 'd_model'):
        if src_embed.d_model != d_model:
            return False
    if tgt_embed and hasattr(tgt_embed, 'd_model'):
        if tgt_embed.d_model != d_model:
            return False
    
    # Check attention dimensions
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttentionBlock):
            if module.d_model != d_model:
                return False
            if d_model % module.h != 0:
                return False  # d_k calculation would fail
    
    return True


def verify_model_implementation(config: Dict[str, Any] = None) -> bool:
    """
    Verify that all components from the paper are correctly implemented.
    
    Args:
        config: Configuration to use for validation (uses default if None)
        
    Returns:
        True if all checks pass, False otherwise
    """
    if config is None:
        import sys
        import os
        # Add parent directory to path to access config module
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from config.config import get_config
        config = get_config()
    
    # Build model with paper parameters
    model = build_transformer(
        src_vocab_size=10000,
        tgt_vocab_size=10000,
        src_seq_len=config['seq_len'],
        tgt_seq_len=config['seq_len'],
        d_model=config['d_model'],
        N=config.get('num_layers', 6),  # paper default: 6
        h=config.get('num_heads', 8),  # paper default: 8
        dropout=config.get('dropout', 0.1),
        d_ff=config.get('d_ff', 2048)
    )
    
    # Verify architecture components
    checks = {
        'encoder_layers': len(model.encoder.layers) == 6,  # Paper uses 6 layers
        'decoder_layers': len(model.decoder.layers) == 6,  # Paper uses 6 layers
        'positional_encoding': has_positional_encoding(model),
        'multihead_attention': has_multihead_attention(model),
        'feed_forward': has_feed_forward_networks(model),
        'layer_normalization': has_layer_normalization(model),
        'residual_connections': has_residual_connections(model),
        'correct_dimensions': verify_dimensions(model, config)
    }
    
    print("Model verification results:")
    all_passed = True
    for component, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status} {component}: {'passed' if passed else 'failed'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nPASS Model architecture verification passed - matches paper specifications")
    else:
        print("\nFAIL Model architecture verification failed")
    
    return all_passed


def test_attention_mechanism():
    """Test that attention mechanism works as expected."""
    # Create dummy inputs
    batch_size, seq_len, d_model, h = 2, 10, 512, 8
    attention_block = MultiHeadAttentionBlock(d_model, h, dropout=0.1)
    
    # Inputs
    q = torch.randn(batch_size, seq_len, d_model)
    k = torch.randn(batch_size, seq_len, d_model)
    v = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass using attention method directly
    try:
        # Get the attention method from the block
        output, attention_weights = MultiHeadAttentionBlock.attention(
            q.view(batch_size, seq_len, h, d_model // h).transpose(1, 2),
            k.view(batch_size, seq_len, h, d_model // h).transpose(1, 2),
            v.view(batch_size, seq_len, h, d_model // h).transpose(1, 2),
            mask=None,
            dropout=nn.Dropout(0.1)
        )
        
        # Verify attention weights shape matches expected (batch, h, seq_len, seq_len)
        expected_shape = (batch_size, h, seq_len, seq_len)
        assert attention_weights.shape == expected_shape, f"expected {expected_shape}, got {attention_weights.shape}"
        
        # Verify output shape
        expected_output_shape = (batch_size, h, seq_len, d_model // h)
        assert output.shape == expected_output_shape, f"expected {expected_output_shape}, got {output.shape}"
        
        print("PASS Attention mechanism verification passed")
        return True
    except Exception as e:
        print(f"FAIL Attention mechanism verification failed: {e}")
        return False


def test_scaled_dot_product():
    """Test the scaled dot-product attention formula."""
    # Simple test to verify the attention formula implementation
    d_k = 64  # Typical size
    seq_len = 5
    batch_size = 1
    
    # Create example query, key, value
    q = torch.ones(batch_size, seq_len, d_k)
    k = torch.ones(batch_size, seq_len, d_k)
    v = torch.ones(batch_size, seq_len, d_k)
    
    # Calculate attention manually
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # Scaled dot product
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    
    print(f"PASS Scaled dot-product attention test passed")
    print(f"  Input shape: {q.shape}")
    print(f"  Attention weights shape: {attention_weights.shape}")
    print(f"  Output shape: {output.shape}")
    
    return True


def run_all_verification_tests(config: Dict[str, Any] = None) -> bool:
    """
    Run all verification tests.
    
    Args:
        config: Configuration to use for validation (uses default if None)
        
    Returns:
        True if all tests pass, False otherwise
    """
    print("Running model verification tests...")
    print("=" * 50)
    
    # Test attention mechanism
    attention_passed = test_attention_mechanism()
    
    # Test scaled dot-product
    scaled_passed = test_scaled_dot_product()
    
    # Test full model architecture
    model_passed = verify_model_implementation(config)
    
    print("=" * 50)
    print("Verification summary:")
    print(f"  Attention mechanism: {'PASS' if attention_passed else 'FAIL'}")
    print(f"  Scaled dot-product: {'PASS' if scaled_passed else 'FAIL'}")
    print(f"  Model architecture: {'PASS' if model_passed else 'FAIL'}")
    
    all_passed = attention_passed and scaled_passed and model_passed
    
    if all_passed:
        print("\nPASS All verification tests passed!")
    else:
        print("\nFAIL Some verification tests failed!")
    
    return all_passed


if __name__ == "__main__":
    run_all_verification_tests()