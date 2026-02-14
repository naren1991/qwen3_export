"""
Optimized CoreML conversion for Qwen3 1.7B

This script converts a Qwen3 model to CoreML with settings that minimize
memory usage during loading on iOS devices.

Key optimizations:
1. compute_precision=FLOAT16 - Keeps computations in fp16
2. Use palettized weights (4-bit) that stay compressed in memory
3. Flexible input shapes for dynamic KV cache
4. Optional: MLState for stateful KV cache (iOS 18+)

Requirements:
    pip install coremltools torch transformers

Usage:
    python convert_qwen3_optimized.py
"""

import torch
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Configuration
MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_NAME = "Qwen3_1_7B_optimized"
MAX_SEQ_LEN = 2048
DEFAULT_SEQ_LEN = 128  # Default/minimum sequence length

def convert_with_torch_export():
    """
    Convert using torch.export (recommended for PyTorch 2.0+)
    This method works with ct.convert() on exported programs.
    """
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    print("Exporting model with torch.export...")
    
    # Create example inputs for tracing
    example_input = torch.randint(0, 1000, (1, DEFAULT_SEQ_LEN), dtype=torch.int32)
    
    # Export the model
    # Use torch.export for cleaner graph
    with torch.no_grad():
        exported_program = torch.export.export(
            model,
            (example_input,),
            dynamic_shapes={
                "input_ids": {1: torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)}
            }
        )
    
    print("Converting to CoreML...")
    
    # Convert with optimizations for low memory loading
    mlmodel = ct.convert(
        exported_program,
        
        # Use float16 compute precision - reduces memory during inference
        compute_precision=ct.precision.FLOAT16,
        
        # Minimum deployment target for best optimizations
        minimum_deployment_target=ct.target.iOS17,
        
        # Convert to ML Program format (required for modern optimizations)
        convert_to="mlprogram",
        
        # Input specifications with flexible shape
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(
                    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=DEFAULT_SEQ_LEN))
                ),
                dtype=np.int32
            )
        ],
        
        # Skip GPU validation to avoid memory issues during conversion
        skip_model_load=True,
    )
    
    print("Applying 4-bit quantization (palettization)...")
    
    # Apply 4-bit weight compression
    # This keeps weights compressed and reduces loading memory
    op_config = ct.optimize.coreml.OpPalettizerConfig(
        mode="kmeans",
        nbits=4,
        weight_threshold=512  # Only quantize weights larger than this
    )
    
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    
    mlmodel_quantized = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
    
    # Save the model
    output_path = f"{OUTPUT_NAME}.mlpackage"
    mlmodel_quantized.save(output_path)
    print(f"✅ Model saved to {output_path}")
    
    return mlmodel_quantized


def convert_with_traced_model():
    """
    Alternative: Convert using torch.jit.trace
    Use this if torch.export doesn't work with your model.
    """
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False  # Disable KV cache for simpler tracing
    )
    model.eval()
    
    # Wrapper to simplify the interface
    class SimpleWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids):
            outputs = self.model(input_ids, use_cache=False)
            return outputs.logits
    
    wrapped_model = SimpleWrapper(model)
    
    print("Tracing model...")
    example_input = torch.randint(0, 1000, (1, DEFAULT_SEQ_LEN), dtype=torch.int32)
    
    with torch.no_grad():
        traced = torch.jit.trace(wrapped_model, example_input)
    
    print("Converting to CoreML...")
    
    mlmodel = ct.convert(
        traced,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(
                    shape=(1, ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=DEFAULT_SEQ_LEN))
                ),
                dtype=np.int32
            )
        ],
    )
    
    print("Applying 4-bit quantization...")
    
    op_config = ct.optimize.coreml.OpPalettizerConfig(
        mode="kmeans",
        nbits=4,
        weight_threshold=512
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel_quantized = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
    
    output_path = f"{OUTPUT_NAME}_traced.mlpackage"
    mlmodel_quantized.save(output_path)
    print(f"✅ Model saved to {output_path}")
    
    return mlmodel_quantized


def convert_with_stateful_kv_cache():
    """
    Advanced: Convert with MLState for stateful KV cache (iOS 18+)
    This provides the most memory-efficient inference.
    
    NOTE: Requires model modifications to externalize KV cache as state.
    """
    print("⚠️ Stateful KV cache conversion requires custom model modifications.")
    print("See: https://apple.github.io/coremltools/docs-guides/source/stateful-models.html")
    
    # This is a template - actual implementation depends on model architecture
    """
    # Example state specification for KV cache:
    
    num_layers = 28  # Qwen3-1.7B has 28 layers
    num_kv_heads = 8
    head_dim = 128
    
    states = []
    for i in range(num_layers):
        # Key cache state
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, num_kv_heads, ct.RangeDim(0, MAX_SEQ_LEN), head_dim),
                    dtype=np.float16
                ),
                name=f"key_cache_{i}"
            )
        )
        # Value cache state
        states.append(
            ct.StateType(
                wrapped_type=ct.TensorType(
                    shape=(1, num_kv_heads, ct.RangeDim(0, MAX_SEQ_LEN), head_dim),
                    dtype=np.float16
                ),
                name=f"value_cache_{i}"
            )
        )
    
    mlmodel = ct.convert(
        exported_program,
        states=states,
        ...
    )
    """
    pass


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen3 1.7B Optimized CoreML Conversion")
    print("=" * 60)
    print()
    print("This will convert the model with settings optimized for")
    print("low memory loading on iOS devices.")
    print()
    print("Options:")
    print("  1. torch.export (recommended for PyTorch 2.0+)")
    print("  2. torch.jit.trace (fallback)")
    print("  3. Stateful KV cache (iOS 18+, advanced)")
    print()
    
    choice = input("Select option (1/2/3): ").strip()
    
    if choice == "1":
        convert_with_torch_export()
    elif choice == "2":
        convert_with_traced_model()
    elif choice == "3":
        convert_with_stateful_kv_cache()
    else:
        print("Invalid option. Running default (torch.export)...")
        convert_with_torch_export()
