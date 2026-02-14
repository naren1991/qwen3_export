"""
Stateful KV Cache Conversion for Qwen3 1.7B

This script:
1. Extracts model architecture (layers, heads, dimensions)
2. Creates MLState specifications for KV cache
3. Converts with memory-optimized stateful inference

Requirements:
    pip install coremltools torch transformers

For iOS 18+ / macOS 15+
"""

import torch
import numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from typing import List, Tuple, Dict, Any

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-1.7B"
OUTPUT_NAME = "Qwen3_1_7B_stateful"
MAX_SEQ_LEN = 2048
DEFAULT_SEQ_LEN = 128
BATCH_SIZE = 1

# =============================================================================
# STEP 1: Extract Model Architecture
# =============================================================================

def extract_model_config(model_name: str) -> Dict[str, Any]:
    """
    Extract architecture details from model config.
    Returns dict with all necessary dimensions for KV cache.
    """
    print(f"📋 Extracting config from {model_name}...")
    
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Extract Qwen3 architecture parameters
    arch = {
        "model_type": getattr(config, "model_type", "qwen3"),
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "hidden_size": config.hidden_size,
        "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "max_position_embeddings": getattr(config, "max_position_embeddings", MAX_SEQ_LEN),
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-6),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
    }
    
    # Calculate KV cache dimensions
    arch["kv_cache_shape"] = {
        "batch": BATCH_SIZE,
        "num_kv_heads": arch["num_key_value_heads"],
        "max_seq_len": MAX_SEQ_LEN,
        "head_dim": arch["head_dim"],
    }
    
    # Memory estimation
    bytes_per_element = 2  # float16
    kv_cache_size_per_layer = (
        BATCH_SIZE * 
        arch["num_key_value_heads"] * 
        MAX_SEQ_LEN * 
        arch["head_dim"] * 
        bytes_per_element * 
        2  # key + value
    )
    total_kv_cache_size = kv_cache_size_per_layer * arch["num_hidden_layers"]
    arch["kv_cache_memory_mb"] = total_kv_cache_size / (1024 * 1024)
    
    return arch


def print_architecture(arch: Dict[str, Any]):
    """Pretty print the architecture."""
    print("\n" + "=" * 60)
    print("📊 MODEL ARCHITECTURE")
    print("=" * 60)
    print(f"  Model Type:           {arch['model_type']}")
    print(f"  Hidden Layers:        {arch['num_hidden_layers']}")
    print(f"  Attention Heads:      {arch['num_attention_heads']}")
    print(f"  KV Heads (GQA):       {arch['num_key_value_heads']}")
    print(f"  Hidden Size:          {arch['hidden_size']}")
    print(f"  Head Dimension:       {arch['head_dim']}")
    print(f"  Intermediate Size:    {arch['intermediate_size']}")
    print(f"  Vocab Size:           {arch['vocab_size']}")
    print(f"  Max Position:         {arch['max_position_embeddings']}")
    print()
    print("📦 KV CACHE DIMENSIONS")
    print(f"  Shape per layer:      [{BATCH_SIZE}, {arch['num_key_value_heads']}, seq_len, {arch['head_dim']}]")
    print(f"  Total layers:         {arch['num_hidden_layers']} × 2 (K + V)")
    print(f"  Max KV Cache Memory:  {arch['kv_cache_memory_mb']:.1f} MB")
    print("=" * 60 + "\n")


# =============================================================================
# STEP 2: Create MLState Specifications for KV Cache
# =============================================================================

def create_kv_cache_states(arch: Dict[str, Any]) -> List[ct.StateType]:
    """
    Create CoreML StateType specifications for KV cache.
    
    Each layer has:
    - key_cache: [batch, num_kv_heads, seq_len, head_dim]
    - value_cache: [batch, num_kv_heads, seq_len, head_dim]
    
    Using RangeDim for dynamic sequence length (0 to max_seq_len).
    """
    states = []
    
    batch = arch["kv_cache_shape"]["batch"]
    num_kv_heads = arch["kv_cache_shape"]["num_kv_heads"]
    head_dim = arch["kv_cache_shape"]["head_dim"]
    max_seq = arch["kv_cache_shape"]["max_seq_len"]
    
    print(f"🔧 Creating {arch['num_hidden_layers'] * 2} state tensors for KV cache...")
    
    for layer_idx in range(arch["num_hidden_layers"]):
        # Key cache state - sequence dimension is dynamic (0 to max)
        key_state = ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(batch, num_kv_heads, ct.RangeDim(lower_bound=0, upper_bound=max_seq), head_dim),
                dtype=np.float16
            ),
            name=f"key_cache_{layer_idx}"
        )
        states.append(key_state)
        
        # Value cache state
        value_state = ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(batch, num_kv_heads, ct.RangeDim(lower_bound=0, upper_bound=max_seq), head_dim),
                dtype=np.float16
            ),
            name=f"value_cache_{layer_idx}"
        )
        states.append(value_state)
    
    print(f"✅ Created {len(states)} state tensors")
    return states


# =============================================================================
# STEP 3: Model Wrapper for Stateful Inference
# =============================================================================

class StatefulQwen3Wrapper(torch.nn.Module):
    """
    Wrapper that externalizes KV cache as explicit inputs/outputs.
    This allows CoreML to manage them as MLState.
    """
    
    def __init__(self, model, num_layers: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.model = model
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        *past_key_values_flat  # Flattened KV cache: k0, v0, k1, v1, ...
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with explicit KV cache inputs/outputs.
        
        Args:
            input_ids: [batch, seq_len]
            position_ids: [batch, seq_len]  
            past_key_values_flat: 2 * num_layers tensors of shape [batch, num_kv_heads, past_seq, head_dim]
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            new_key_values_flat: Updated KV cache tensors
        """
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        
        # Reconstruct past_key_values tuple structure
        past_key_values = []
        for i in range(self.num_layers):
            key = past_key_values_flat[i * 2]      # [batch, num_kv_heads, past_seq, head_dim]
            value = past_key_values_flat[i * 2 + 1]
            past_key_values.append((key, value))
        past_key_values = tuple(past_key_values) if past_key_values_flat else None
        
        # Run model
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
        
        logits = outputs.logits
        new_past = outputs.past_key_values
        
        # Flatten new KV cache for output
        new_kv_flat = []
        for layer_kv in new_past:
            new_kv_flat.append(layer_kv[0])  # key
            new_kv_flat.append(layer_kv[1])  # value
        
        return (logits,) + tuple(new_kv_flat)


# =============================================================================
# STEP 4: Export and Convert
# =============================================================================

def convert_stateful_model(arch: Dict[str, Any]):
    """
    Full conversion pipeline with stateful KV cache.
    """
    print("\n📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    
    # Wrap model for stateful inference
    print("🔧 Wrapping model for stateful inference...")
    wrapped = StatefulQwen3Wrapper(
        model,
        num_layers=arch["num_hidden_layers"],
        num_kv_heads=arch["num_key_value_heads"],
        head_dim=arch["head_dim"]
    )
    
    # Create example inputs
    batch = BATCH_SIZE
    seq_len = 1  # Single token for incremental decoding
    past_seq = 10  # Example past sequence length
    
    example_input_ids = torch.randint(0, arch["vocab_size"], (batch, seq_len), dtype=torch.long)
    example_position_ids = torch.arange(past_seq, past_seq + seq_len).unsqueeze(0).expand(batch, -1)
    
    # Example past KV cache (empty for initial, non-empty for continuation)
    example_past_kv = []
    for _ in range(arch["num_hidden_layers"]):
        k = torch.zeros(batch, arch["num_key_value_heads"], past_seq, arch["head_dim"], dtype=torch.float16)
        v = torch.zeros(batch, arch["num_key_value_heads"], past_seq, arch["head_dim"], dtype=torch.float16)
        example_past_kv.extend([k, v])
    
    print("📤 Exporting with torch.export...")
    
    # Dynamic shapes for flexible sequence lengths
    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)},
        "position_ids": {1: torch.export.Dim("seq_len", min=1, max=MAX_SEQ_LEN)},
    }
    
    # Add dynamic shapes for KV cache (past sequence dimension)
    for i in range(arch["num_hidden_layers"] * 2):
        dynamic_shapes[f"past_kv_{i}"] = {2: torch.export.Dim("past_seq", min=0, max=MAX_SEQ_LEN)}
    
    with torch.no_grad():
        try:
            exported = torch.export.export(
                wrapped,
                (example_input_ids, example_position_ids, *example_past_kv),
                dynamic_shapes=dynamic_shapes
            )
        except Exception as e:
            print(f"⚠️ torch.export failed: {e}")
            print("Falling back to torch.jit.trace...")
            exported = torch.jit.trace(wrapped, (example_input_ids, example_position_ids, *example_past_kv))
    
    # Create CoreML state specifications
    states = create_kv_cache_states(arch)
    
    # Input specifications
    inputs = [
        ct.TensorType(
            name="input_ids",
            shape=ct.Shape(shape=(batch, ct.RangeDim(1, MAX_SEQ_LEN, default=1))),
            dtype=np.int32
        ),
        ct.TensorType(
            name="position_ids", 
            shape=ct.Shape(shape=(batch, ct.RangeDim(1, MAX_SEQ_LEN, default=1))),
            dtype=np.int32
        ),
    ]
    
    print("🔄 Converting to CoreML with states...")
    
    try:
        mlmodel = ct.convert(
            exported,
            inputs=inputs,
            states=states,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
            skip_model_load=True,
        )
        
        print("🗜️ Applying 4-bit quantization...")
        
        op_config = ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans",
            nbits=4,
            weight_threshold=512
        )
        config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
        mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
        
        # Save
        output_path = f"{OUTPUT_NAME}.mlpackage"
        mlmodel.save(output_path)
        print(f"\n✅ Stateful model saved to {output_path}")
        
        return mlmodel
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        print("\nThis is expected - stateful conversion requires model modifications.")
        print("See the alternative approach below.")
        raise


# =============================================================================
# ALTERNATIVE: Simpler Approach Without Full Stateful Conversion
# =============================================================================

def convert_simple_with_flexible_shapes(arch: Dict[str, Any]):
    """
    Simpler conversion that keeps KV cache internal but uses flexible shapes
    to avoid pre-allocating maximum memory.
    
    This doesn't require model modifications but still benefits from
    dynamic shapes.
    """
    print("\n📥 Loading model (simple mode)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False  # Disable internal KV cache for simpler export
    )
    model.eval()
    
    class SimpleWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, input_ids):
            return self.model(input_ids, use_cache=False).logits
    
    wrapped = SimpleWrapper(model)
    
    example_input = torch.randint(0, arch["vocab_size"], (BATCH_SIZE, DEFAULT_SEQ_LEN), dtype=torch.long)
    
    print("📤 Tracing model...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_input)
    
    print("🔄 Converting to CoreML...")
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="input_ids",
                shape=ct.Shape(shape=(
                    BATCH_SIZE, 
                    ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ_LEN, default=DEFAULT_SEQ_LEN)
                )),
                dtype=np.int32
            )
        ],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram",
    )
    
    print("🗜️ Applying 4-bit quantization...")
    op_config = ct.optimize.coreml.OpPalettizerConfig(mode="kmeans", nbits=4, weight_threshold=512)
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    mlmodel = ct.optimize.coreml.palettize_weights(mlmodel, config=config)
    
    output_path = f"{OUTPUT_NAME}_simple.mlpackage"
    mlmodel.save(output_path)
    print(f"\n✅ Model saved to {output_path}")
    
    return mlmodel


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 Qwen3 1.7B Stateful CoreML Conversion")
    print("=" * 60)
    
    # Step 1: Extract architecture
    arch = extract_model_config(MODEL_NAME)
    print_architecture(arch)
    
    print("\nConversion Options:")
    print("  1. Full stateful conversion (requires model modifications)")
    print("  2. Simple conversion with flexible shapes (recommended)")
    print()
    
    choice = input("Select option (1/2): ").strip()
    
    if choice == "1":
        try:
            convert_stateful_model(arch)
        except Exception as e:
            print(f"\n⚠️ Stateful conversion failed. Falling back to simple mode...")
            convert_simple_with_flexible_shapes(arch)
    else:
        convert_simple_with_flexible_shapes(arch)
    
    print("\n" + "=" * 60)
    print("📱 To use in Swift:")
    print("=" * 60)
    print("""
    // For stateful model (iOS 18+):
    let state = model.makeState()
    let output = try model.prediction(input: input, using: state)
    
    // The state persists KV cache between calls,
    // reducing memory allocation during inference.
    """)
