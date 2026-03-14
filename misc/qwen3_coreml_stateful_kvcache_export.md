# Qwen3 1.7B — Stateful KV-Cache Core ML Export: All Approaches

> **Issue**: #47 — Investigate KV cache implementation and impact on performance
> **Model**: Qwen/Qwen3-1.7B (int4 quantized, Core ML)
> **Target**: iOS 18+ / macOS 15+ (MLState API)
> **Max Context**: 2048 tokens

---

## Table of Contents

1. [Model Specifications](#1-model-specifications)
2. [Shared Infrastructure](#2-shared-infrastructure)
3. [Approach A: Wrapper + torch.jit.trace (Mistral-style)](#3-approach-a-wrapper--torchjiittrace-mistral-style)
4. [Approach B: Wrapper + torch.export](#4-approach-b-wrapper--torchexport)
5. [Approach C: KV-Cache-as-I/O (Explicit Inputs/Outputs)](#5-approach-c-kv-cache-as-io-explicit-inputsoutputs)
6. [Approach D: HuggingFace StaticCache + torch.export](#6-approach-d-huggingface-staticcache--torchexport)
7. [Quantization (Shared)](#7-quantization-shared)
8. [Verification & Benchmarking](#8-verification--benchmarking)
9. [Comparison Matrix](#9-comparison-matrix)

---

## 1. Model Specifications

### Qwen3 1.7B Architecture

| Parameter | Value |
|---|---|
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 4 (GQA, 4:1 ratio) |
| `hidden_size` | 2048 |
| `head_dim` | 128 |
| `vocab_size` | 151936 |
| `max_position_embeddings` | 40960 (we use 2048) |
| `rms_norm_eps` | 1e-6 |
| `sliding_window` | Per-layer (layer_types config) |

### KV Cache Memory Budget

```
Per layer (one of K or V): 1 × 4 × 2048 × 128 × 2 bytes = 2,097,152 bytes ≈ 2 MB
Both K and V per layer: ~4 MB
All 28 layers: 28 × 4 MB ≈ 112 MB (FP16)
```

With Apple's stateful buffers, this 112 MB lives on-GPU and is updated in-place — no copies per step.

### Qwen3-Specific Attention Features

Unlike Mistral, Qwen3 has:
- **QK Norm**: `Qwen3RMSNorm` applied to Q and K per-head **before** RoPE
- **No `ATTENTION_CLASSES` dict**: Uses `ALL_ATTENTION_FUNCTIONS.get_interface()` dispatch
- **Single `Qwen3Attention` class**: No separate SDPA/Flash subclasses
- **Sliding window**: Some layers may use sliding window attention (per `config.layer_types`)

---

## 2. Shared Infrastructure

### 2.1 Environment Setup

```python
# Verified working environment
# PyTorch 2.6, transformers 4.51, coremltools 8.0

import torch
import numpy as np
import coremltools as ct
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.cache_utils import Cache

print(f"PyTorch: {torch.__version__}")
print(f"coremltools: {ct.__version__}")
```

### 2.2 Configuration Constants

```python
MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_CONTEXT_SIZE = 2048
BATCH_SIZE = 1
DTYPE = torch.float16

config = AutoConfig.from_pretrained(MODEL_ID)

NUM_LAYERS = config.num_hidden_layers        # 28
NUM_KV_HEADS = config.num_key_value_heads    # 4
NUM_ATTN_HEADS = config.num_attention_heads  # 16
HEAD_DIM = config.hidden_size // config.num_attention_heads  # 128
HIDDEN_SIZE = config.hidden_size             # 2048
VOCAB_SIZE = config.vocab_size               # 151936
RMS_NORM_EPS = config.rms_norm_eps           # 1e-6

print(f"Layers: {NUM_LAYERS}, KV Heads: {NUM_KV_HEADS}, Head Dim: {HEAD_DIM}")
print(f"KV cache shape per layer: ({BATCH_SIZE}, {NUM_KV_HEADS}, {MAX_CONTEXT_SIZE}, {HEAD_DIM})")
```

### 2.3 Load Base Model

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    attn_implementation="eager",  # Required for tracing/export
    low_cpu_mem_usage=True,
)
base_model = base_model.to(DTYPE).cpu().eval()
base_model.config.use_cache = False  # Will override per-approach

print(f"Model loaded. Parameters: {sum(p.numel() for p in base_model.parameters()):,}")
```

### 2.4 Baseline Stateless Output (for verification)

```python
# Generate reference output for numerical verification
test_prompt = "Hello, how are you"
test_tokens = tokenizer(test_prompt, return_tensors="pt")["input_ids"]

with torch.no_grad():
    base_model.config.use_cache = False
    ref_output = base_model(test_tokens).logits
    ref_last_logits = ref_output[0, -1, :].float().numpy()

print(f"Reference logits shape: {ref_output.shape}")
print(f"Top-5 token IDs: {np.argsort(ref_last_logits)[-5:][::-1]}")
```

### 2.5 Causal Mask Builder Utility

```python
def build_causal_mask(q_len: int, kv_len: int, dtype=np.float16) -> np.ndarray:
    """
    Build a causal attention mask.
    Shape: (1, 1, q_len, kv_len) — broadcastable over batch and heads.
    0.0 = attend, -inf (or large negative) = mask out.
    """
    mask = np.full((1, 1, q_len, kv_len), -1e4, dtype=dtype)
    for i in range(q_len):
        # Attend to all positions up to and including the current position
        start = kv_len - q_len  # offset for past KV
        for j in range(start + i + 1):
            mask[0, 0, i, j] = 0.0
    return mask


def build_causal_mask_torch(q_len: int, kv_len: int, dtype=torch.float16) -> torch.Tensor:
    """Torch version of causal mask for tracing."""
    mask = torch.full((1, 1, q_len, kv_len), -1e4, dtype=dtype)
    for i in range(q_len):
        start = kv_len - q_len
        for j in range(start + i + 1):
            mask[0, 0, i, j] = 0.0
    return mask


# Verify
prefill_mask = build_causal_mask(4, 4)
print(f"Prefill mask shape: {prefill_mask.shape}")
print(prefill_mask[0, 0])
# Expected: upper triangle is -1e4, lower triangle (including diagonal) is 0
```

---

## 3. Approach A: Wrapper + `torch.jit.trace` (Mistral-style)

> **Reference**: [HuggingFace Mistral7B export.py](https://github.com/huggingface/swift-transformers/blob/preview/Examples/Mistral7B/export.py)
> **Status**: Proven pattern (Apple WWDC24 + HuggingFace)

This is the official approach demonstrated by Apple and HuggingFace for Mistral 7B.
The key idea: create a wrapper `nn.Module` that (1) overrides the attention class to do slice-update KV caching,
(2) registers the KV cache as `register_buffer` so `ct.convert` can capture them as `StateType`.

### 3.1 SliceUpdateKeyValueCache

```python
from typing import Tuple, Optional

class SliceUpdateKeyValueCache(Cache):
    """
    KV cache that supports in-place slice updates.
    Shape: (num_layers, batch_size, num_kv_heads, max_context_size, head_dim)
    
    Instead of concatenating new KV states (DynamicCache style),
    we pre-allocate the full buffer and write into slices.
    This is required for Core ML stateful buffers.
    """
    
    def __init__(
        self,
        shape: Tuple[int, ...],
        device: str = "cpu",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        self.past_seen_tokens: int = 0
        self.k_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
        self.v_cache: torch.Tensor = torch.zeros(shape, dtype=dtype, device=device)
    
    def update(
        self,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        layer_idx: int,
        slice_indices: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update key/value cache for slice [begin, end).
        Return accumulated key/value from [0, end).
        
        k_state shape: (batch, num_kv_heads, q_len, head_dim)
        """
        if len(slice_indices) != 2:
            raise ValueError(f"Expected (begin, end), got {slice_indices}")
        
        begin, end = slice_indices
        
        # In-place slice write
        self.k_cache[layer_idx, :, :k_state.shape[1], begin:end, :] = k_state
        self.v_cache[layer_idx, :, :v_state.shape[1], begin:end, :] = v_state
        
        # Return accumulated cache up to end
        k_out = self.k_cache[layer_idx, :, :, :end, :]
        v_out = self.v_cache[layer_idx, :, :, :end, :]
        
        return k_out, v_out
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self.past_seen_tokens
```

### 3.2 Verify SliceUpdateKeyValueCache independently

```python
# Sanity check the cache
cache = SliceUpdateKeyValueCache(
    shape=(NUM_LAYERS, BATCH_SIZE, NUM_KV_HEADS, MAX_CONTEXT_SIZE, HEAD_DIM),
    dtype=DTYPE,
)

# Simulate prefill of 5 tokens at layer 0
fake_k = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM, dtype=DTYPE)
fake_v = torch.randn(1, NUM_KV_HEADS, 5, HEAD_DIM, dtype=DTYPE)
k_out, v_out = cache.update(fake_k, fake_v, layer_idx=0, slice_indices=(0, 5))

assert k_out.shape == (1, NUM_KV_HEADS, 5, HEAD_DIM), f"Expected (1,4,5,128), got {k_out.shape}"
assert torch.allclose(k_out[0, :, :5, :], fake_k[0]), "Prefill K mismatch"

# Simulate decode step 6 at layer 0
fake_k2 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
fake_v2 = torch.randn(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
k_out2, v_out2 = cache.update(fake_k2, fake_v2, layer_idx=0, slice_indices=(5, 6))

assert k_out2.shape == (1, NUM_KV_HEADS, 6, HEAD_DIM), f"Expected (1,4,6,128), got {k_out2.shape}"
assert torch.allclose(k_out2[0, :, :5, :], fake_k[0]), "Previous K shouldn't change"
assert torch.allclose(k_out2[0, :, 5:6, :], fake_k2[0]), "New K should be at position 5"

print("✅ SliceUpdateKeyValueCache passed all checks")
```

### 3.3 SliceUpdateQwen3Attention

```python
import math
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3RMSNorm,
)

# For older transformers that may use modeling_utils repeat_kv
try:
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
except ImportError:
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

try:
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
except ImportError:
    from transformers.modeling_rope_utils import apply_rotary_pos_emb


class SliceUpdateQwen3Attention(Qwen3Attention):
    """
    Qwen3 attention with in-place KV cache slice updates.
    
    Key differences from Mistral version:
    - Qwen3 has QK Norm (RMSNorm on Q and K per-head, before RoPE)
    - Uses ALL_ATTENTION_FUNCTIONS dispatch (we bypass it, use SDPA directly)
    - GQA ratio: 16 query heads / 4 KV heads = 4:1
    """
    
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        bsz, q_len, _ = hidden_states.size()
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape: (batch, seq, hidden) -> (batch, heads, seq, head_dim)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        # QK Norm (Qwen3-specific) — RMSNorm per head before RoPE
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        # RoPE
        # position_ids is passed through kwargs in some versions
        if position_ids is None:
            position_ids = kwargs.get("position_ids", None)
        
        # Get rotary embeddings
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self.rotary_emb(value_states, position_ids)
        
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        
        # Slice-update KV cache
        end_step = attention_mask.shape[-1]
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            self.layer_idx,
            slice_indices=(end_step - q_len, end_step),
        )
        
        # GQA: expand KV heads to match query heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # Scaled dot-product attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
        )
        
        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, None
```

### 3.4 Verify attention shapes

```python
# Quick shape verification for the custom attention
print(f"num_heads: {NUM_ATTN_HEADS}")
print(f"num_kv_heads: {NUM_KV_HEADS}")
print(f"num_key_value_groups: {NUM_ATTN_HEADS // NUM_KV_HEADS}")  # 4
print(f"head_dim: {HEAD_DIM}")

# Check that q_norm and k_norm exist on the original attention
sample_attn = base_model.model.layers[0].self_attn
print(f"Has q_norm: {hasattr(sample_attn, 'q_norm')}")
print(f"Has k_norm: {hasattr(sample_attn, 'k_norm')}")
print(f"q_norm type: {type(sample_attn.q_norm)}")
print(f"q_proj shape: {sample_attn.q_proj.weight.shape}")  # (2048, 2048)
print(f"k_proj shape: {sample_attn.k_proj.weight.shape}")  # (512, 2048)
print(f"v_proj shape: {sample_attn.v_proj.weight.shape}")  # (512, 2048)
print(f"o_proj shape: {sample_attn.o_proj.weight.shape}")  # (2048, 2048)
```

### 3.5 StatefulQwen3ForCausalLM Wrapper

```python
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Attention,
)
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config


class StatefulQwen3ForCausalLM(torch.nn.Module):
    """
    Wrapper around Qwen3ForCausalLM that:
    1. Replaces attention modules with SliceUpdateQwen3Attention
    2. Registers KV cache buffers for Core ML StateType
    3. Provides a trace-friendly forward() signature
    """
    
    def __init__(
        self,
        model_path: str,
        max_context_size: int = 2048,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        
        # Replace attention classes before loading
        # Qwen3 doesn't have ATTENTION_CLASSES dict, so we monkey-patch
        # Save original class
        original_attn_class = Qwen3Attention
        
        # Temporarily replace the class in the module
        import transformers.models.qwen3.modeling_qwen3 as qwen3_module
        qwen3_module.Qwen3Attention = SliceUpdateQwen3Attention
        
        # Load model with patched attention
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        # Restore original class
        qwen3_module.Qwen3Attention = original_attn_class
        
        # Verify all attention modules are SliceUpdateQwen3Attention
        for name, module in self.model.named_modules():
            if isinstance(module, SliceUpdateQwen3Attention):
                pass  # Good
            elif isinstance(module, Qwen3Attention) and not isinstance(module, SliceUpdateQwen3Attention):
                print(f"WARNING: {name} is still original Qwen3Attention!")
        
        # Setup KV cache
        config: Qwen3Config = self.model.config
        self.kv_cache_shape: Tuple[int, ...] = (
            config.num_hidden_layers,     # 28
            batch_size,                    # 1
            config.num_key_value_heads,    # 4
            max_context_size,              # 2048
            config.hidden_size // config.num_attention_heads,  # 128
        )
        
        self.kv_cache = SliceUpdateKeyValueCache(
            shape=self.kv_cache_shape,
            dtype=DTYPE,
        )
        
        # Register buffers — these become Core ML StateType
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — token IDs
            causal_mask: (1, 1, seq_len, kv_len) — attention mask
                kv_len = past_seen_tokens + seq_len
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        # Compute how many tokens are already in the KV cache
        self.kv_cache.past_seen_tokens = (
            causal_mask.shape[-1] - input_ids.shape[-1]
        )
        
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits
```

### 3.6 Verify the wrapper loads correctly

```python
# Test instantiation
print("Loading StatefulQwen3ForCausalLM...")
stateful_model = StatefulQwen3ForCausalLM(
    MODEL_ID,
    max_context_size=MAX_CONTEXT_SIZE,
)
stateful_model.eval()

print(f"KV cache shape: {stateful_model.kv_cache_shape}")
print(f"keyCache buffer shape: {stateful_model.keyCache.shape}")
print(f"valueCache buffer shape: {stateful_model.valueCache.shape}")

# Count patched attention modules
patched_count = sum(
    1 for m in stateful_model.model.modules()
    if isinstance(m, SliceUpdateQwen3Attention)
)
print(f"Patched attention modules: {patched_count} (expected {NUM_LAYERS})")
assert patched_count == NUM_LAYERS, "Not all layers were patched!"

# Quick forward pass test (prefill 4 tokens)
test_ids = torch.zeros((1, 4), dtype=torch.int32)
test_mask = build_causal_mask_torch(4, 4, dtype=DTYPE)

with torch.no_grad():
    logits = stateful_model(test_ids, test_mask)
    print(f"Prefill output shape: {logits.shape}")  # (1, 4, 151936)

print("✅ StatefulQwen3ForCausalLM instantiation and forward pass OK")
```

### 3.7 Trace the stateful model

```python
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("coremltools").setLevel(logging.ERROR)

# Trace inputs — small sequence for tracing, ct.RangeDim handles flexibility
trace_input_ids = torch.zeros((1, 2), dtype=torch.int32)
trace_causal_mask = torch.zeros((1, 1, 2, 5), dtype=DTYPE)

print("Tracing with torch.jit.trace...")
traced_model = torch.jit.trace(
    stateful_model,
    [trace_input_ids, trace_causal_mask],
)
print(f"✅ Traced successfully")

# Optional: inspect the trace graph
# print(traced_model.graph)
```

### 3.8 Verify traced model numerics

```python
# Reset KV cache and compare traced vs eager
stateful_model.kv_cache.k_cache.zero_()
stateful_model.kv_cache.v_cache.zero_()
stateful_model.kv_cache.past_seen_tokens = 0

verify_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int32)
verify_mask = build_causal_mask_torch(4, 4, dtype=DTYPE)

with torch.no_grad():
    eager_out = stateful_model(verify_ids, verify_mask)

# Reset cache again
stateful_model.kv_cache.k_cache.zero_()
stateful_model.kv_cache.v_cache.zero_()
stateful_model.kv_cache.past_seen_tokens = 0

with torch.no_grad():
    traced_out = traced_model(verify_ids, verify_mask)

diff = (eager_out - traced_out).abs().max().item()
print(f"Max absolute difference (eager vs traced): {diff}")
assert diff < 1e-3, f"Numerical mismatch too large: {diff}"
print("✅ Traced model matches eager model")
```

### 3.9 Convert to Core ML with states

```python
# Save kv_cache_shape before deleting the model
kv_cache_shape = stateful_model.kv_cache_shape
del stateful_model  # Free memory

# Define flexible input dimensions
query_length = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)
end_step_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)

inputs = [
    ct.TensorType(
        shape=(1, query_length),
        dtype=np.int32,
        name="inputIds",
    ),
    ct.TensorType(
        shape=(1, 1, query_length, end_step_dim),
        dtype=np.float16,
        name="causalMask",
    ),
]

outputs = [ct.TensorType(dtype=np.float16, name="logits")]

# Define states — names MUST match register_buffer names
states = [
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=kv_cache_shape,
            dtype=np.float16,
        ),
        name="keyCache",
    ),
    ct.StateType(
        wrapped_type=ct.TensorType(
            shape=kv_cache_shape,
            dtype=np.float16,
        ),
        name="valueCache",
    ),
]

print("Converting to Core ML...")
mlmodel_a = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    states=states,
    minimum_deployment_target=ct.target.iOS18,
    skip_model_load=True,
)

print("✅ Core ML conversion complete (Approach A)")
```

### 3.10 Save and verify Core ML spec

```python
OUTPUT_PATH_A = "models/Qwen3_1_7B_stateful_trace_fp16.mlpackage"
mlmodel_a.save(OUTPUT_PATH_A)
print(f"Saved to {OUTPUT_PATH_A}")

# Inspect spec
spec = mlmodel_a.get_spec()

print("\n=== Inputs ===")
for inp in spec.description.input:
    print(f"  {inp.name}: {inp.type.WhichOneof('Type')}")

print("\n=== Outputs ===")
for out in spec.description.output:
    print(f"  {out.name}: {out.type.WhichOneof('Type')}")

print("\n=== States ===")
for state in spec.description.state:
    print(f"  {state.name}")

del traced_model
```

---

## 4. Approach B: Wrapper + `torch.export`

> **Rationale**: `torch.export` is the modern replacement for `torch.jit.trace`.
> It already works for the stateless Qwen3 model in the existing notebook.
> The question is whether it correctly captures `register_buffer` mutations.

### 4.1 Why torch.export may work

`torch.export` in PyTorch 2.6 captures buffer mutations as graph mutations.
When a buffer is modified in-place (slice assignment), the exported program's
graph records these mutations. `coremltools 8.0` maps graph mutations on
registered buffers to Core ML state read/write operations when `states=` is provided.

### 4.2 Stateful model for export (modified forward signature)

```python
class StatefulQwen3ForExport(torch.nn.Module):
    """
    Same as StatefulQwen3ForCausalLM but with a forward() signature
    compatible with torch.export (no **kwargs, explicit shapes).
    """
    
    def __init__(
        self,
        model_path: str,
        max_context_size: int = 2048,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        
        # Monkey-patch attention
        import transformers.models.qwen3.modeling_qwen3 as qwen3_module
        original_attn_class = qwen3_module.Qwen3Attention
        qwen3_module.Qwen3Attention = SliceUpdateQwen3Attention
        
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        qwen3_module.Qwen3Attention = original_attn_class
        
        config: Qwen3Config = self.model.config
        self.kv_cache_shape = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
        
        self.kv_cache = SliceUpdateKeyValueCache(
            shape=self.kv_cache_shape,
            dtype=DTYPE,
        )
        
        self.register_buffer("keyCache", self.kv_cache.k_cache)
        self.register_buffer("valueCache", self.kv_cache.v_cache)
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        self.kv_cache.past_seen_tokens = (
            causal_mask.shape[-1] - input_ids.shape[-1]
        )
        return self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits
```

### 4.3 Export with torch.export

```python
stateful_export_model = StatefulQwen3ForExport(
    MODEL_ID,
    max_context_size=MAX_CONTEXT_SIZE,
)
stateful_export_model.eval()

# Example inputs for export
example_input_ids = torch.zeros((1, 32), dtype=torch.long)
example_mask = build_causal_mask_torch(32, 32, dtype=DTYPE)

from torch.export import export, Dim

seq_dim = Dim("seq_len", min=1, max=MAX_CONTEXT_SIZE)
kv_dim = Dim("kv_len", min=1, max=MAX_CONTEXT_SIZE)

print("Running torch.export.export()...")
try:
    exported_program = export(
        stateful_export_model,
        (example_input_ids, example_mask),
        dynamic_shapes={
            "input_ids": {1: seq_dim},
            "causal_mask": {2: seq_dim, 3: kv_dim},
        },
    )
    exported_program = exported_program.run_decompositions({})
    print("✅ torch.export succeeded")
    
    # Check for buffer mutations in the graph
    graph_str = str(exported_program.graph_module.graph)
    has_mutations = "mutate" in graph_str.lower() or "buffer" in graph_str.lower()
    print(f"Graph has buffer-related ops: {has_mutations}")
    
except Exception as e:
    print(f"❌ torch.export failed: {e}")
    print("This may be due to in-place buffer mutations not being supported.")
    print("Fallback: Use Approach A (torch.jit.trace) instead.")
    exported_program = None
```

### 4.4 Convert exported program to Core ML

```python
if exported_program is not None:
    kv_cache_shape_b = stateful_export_model.kv_cache_shape
    del stateful_export_model
    
    states_b = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=kv_cache_shape_b,
                dtype=np.float16,
            ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=kv_cache_shape_b,
                dtype=np.float16,
            ),
            name="valueCache",
        ),
    ]
    
    print("Converting exported program to Core ML...")
    try:
        mlmodel_b = ct.convert(
            exported_program,
            convert_to="mlprogram",
            states=states_b,
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )
        mlmodel_b.save("models/Qwen3_1_7B_stateful_export_fp16.mlpackage")
        print("✅ Core ML conversion complete (Approach B)")
        
    except Exception as e:
        print(f"❌ Core ML conversion failed: {e}")
        print("The export graph may not have buffer mutations in a form coremltools expects.")
        mlmodel_b = None
else:
    mlmodel_b = None
    print("Skipping Core ML conversion (export failed)")
```

---

## 5. Approach C: KV-Cache-as-I/O (Explicit Inputs/Outputs)

> **Rationale**: Entirely skip MLState. Make KV caches explicit model I/O.
> Simpler, guaranteed to work, but ~13x slower than stateful approach (per Apple benchmarks)
> because cache data must be copied to/from the model each step.

### 5.1 Qwen3 with explicit KV cache I/O

```python
class Qwen3WithExplicitKVCache(torch.nn.Module):
    """
    Qwen3 wrapper where the KV cache is an explicit input/output pair.
    
    Inputs:  input_ids, causal_mask, key_cache_in, value_cache_in
    Outputs: logits, key_cache_out, value_cache_out
    
    No register_buffer, no MLState — pure functional.
    """
    
    def __init__(
        self,
        model_path: str,
        max_context_size: int = 2048,
        batch_size: int = 1,
    ) -> None:
        super().__init__()
        
        import transformers.models.qwen3.modeling_qwen3 as qwen3_module
        original_attn_class = qwen3_module.Qwen3Attention
        qwen3_module.Qwen3Attention = SliceUpdateQwen3Attention
        
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        qwen3_module.Qwen3Attention = original_attn_class
        
        config = self.model.config
        self.kv_cache_shape = (
            config.num_hidden_layers,
            batch_size,
            config.num_key_value_heads,
            max_context_size,
            config.hidden_size // config.num_attention_heads,
        )
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
        key_cache_in: torch.Tensor,
        value_cache_in: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (logits, key_cache_out, value_cache_out).
        The caller manages the cache externally.
        """
        # Create cache wrapper around the input tensors
        cache = SliceUpdateKeyValueCache.__new__(SliceUpdateKeyValueCache)
        Cache.__init__(cache)
        cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        cache.k_cache = key_cache_in
        cache.v_cache = value_cache_in
        
        logits = self.model(
            input_ids,
            attention_mask=causal_mask,
            past_key_values=cache,
            use_cache=True,
        ).logits
        
        return logits, cache.k_cache, cache.v_cache
```

### 5.2 Trace the explicit-I/O model

```python
explicit_model = Qwen3WithExplicitKVCache(
    MODEL_ID,
    max_context_size=MAX_CONTEXT_SIZE,
)
explicit_model.eval()

# Trace inputs
trace_ids = torch.zeros((1, 2), dtype=torch.int32)
trace_mask = torch.zeros((1, 1, 2, 5), dtype=DTYPE)
trace_k = torch.zeros(explicit_model.kv_cache_shape, dtype=DTYPE)
trace_v = torch.zeros(explicit_model.kv_cache_shape, dtype=DTYPE)

print("Tracing explicit KV cache model...")
traced_explicit = torch.jit.trace(
    explicit_model,
    [trace_ids, trace_mask, trace_k, trace_v],
)
print("✅ Traced successfully")
```

### 5.3 Convert to Core ML (no states)

```python
kv_shape = explicit_model.kv_cache_shape
del explicit_model

query_length_c = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)
end_step_dim_c = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)

inputs_c = [
    ct.TensorType(shape=(1, query_length_c), dtype=np.int32, name="inputIds"),
    ct.TensorType(
        shape=(1, 1, query_length_c, end_step_dim_c),
        dtype=np.float16,
        name="causalMask",
    ),
    ct.TensorType(shape=kv_shape, dtype=np.float16, name="keyCacheIn"),
    ct.TensorType(shape=kv_shape, dtype=np.float16, name="valueCacheIn"),
]

outputs_c = [
    ct.TensorType(dtype=np.float16, name="logits"),
    ct.TensorType(dtype=np.float16, name="keyCacheOut"),
    ct.TensorType(dtype=np.float16, name="valueCacheOut"),
]

print("Converting to Core ML (no states, explicit I/O)...")
mlmodel_c = ct.convert(
    traced_explicit,
    inputs=inputs_c,
    outputs=outputs_c,
    minimum_deployment_target=ct.target.iOS18,
    skip_model_load=True,
)

mlmodel_c.save("models/Qwen3_1_7B_explicit_kv_fp16.mlpackage")
print("✅ Core ML conversion complete (Approach C)")

del traced_explicit
```

### 5.4 Inference loop for explicit I/O approach (Python reference)

```python
"""
Reference inference loop for Approach C.
In the iOS app, this would be in Swift with MLMultiArray.

NOTE: This approach copies ~112 MB of KV cache data every forward pass.
Apple's benchmarks show stateful approach is ~13x faster.
"""

def generate_explicit_kv(mlmodel, tokenizer, prompt, max_new_tokens=50):
    tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
    seq_len = tokens.shape[1]
    
    # Initialize empty KV cache
    k_cache = np.zeros(kv_shape, dtype=np.float16)
    v_cache = np.zeros(kv_shape, dtype=np.float16)
    
    generated = list(tokens[0])
    
    # Prefill
    mask = build_causal_mask(seq_len, seq_len)
    result = mlmodel.predict({
        "inputIds": tokens,
        "causalMask": mask,
        "keyCacheIn": k_cache,
        "valueCacheIn": v_cache,
    })
    k_cache = result["keyCacheOut"]
    v_cache = result["valueCacheOut"]
    next_token = int(np.argmax(result["logits"][0, -1, :]))
    generated.append(next_token)
    
    # Decode loop
    for step in range(max_new_tokens - 1):
        past_len = seq_len + step + 1
        token_input = np.array([[next_token]], dtype=np.int32)
        mask = build_causal_mask(1, past_len)
        
        result = mlmodel.predict({
            "inputIds": token_input,
            "causalMask": mask,
            "keyCacheIn": k_cache,
            "valueCacheIn": v_cache,
        })
        k_cache = result["keyCacheOut"]
        v_cache = result["valueCacheOut"]
        next_token = int(np.argmax(result["logits"][0, -1, :]))
        generated.append(next_token)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated)
```

---

## 6. Approach D: HuggingFace `StaticCache` + `torch.export`

> **Rationale**: Use HuggingFace's built-in `StaticCache` which pre-allocates
> fixed-size KV cache tensors. This is designed for `torch.compile` and `torch.export`.
> Combined with the `register_buffer` pattern, it might integrate with Core ML states.

### 6.1 Understanding StaticCache

```python
from transformers.cache_utils import StaticCache

"""
StaticCache pre-allocates tensors of shape:
  (batch_size, num_heads, max_cache_len, head_dim)
per layer, and mutates them in-place via index assignment.

This IS compatible with torch.export and torch.compile.
The question is: can we combine it with register_buffer + ct.StateType?
"""

# Create a StaticCache for our model config
static_cache = StaticCache(
    config=base_model.config,
    max_cache_len=MAX_CONTEXT_SIZE,
)

print(f"Number of cache layers: {len(static_cache._cache_layers)}")
layer0 = static_cache._cache_layers[0]
print(f"Layer 0 key shape: {layer0.key_cache.shape if hasattr(layer0, 'key_cache') else 'not initialized (lazy)'}")
print(f"Type: {type(layer0)}")
```

### 6.2 StaticCache wrapper with register_buffer

```python
class StaticCacheQwen3Wrapper(torch.nn.Module):
    """
    Uses HuggingFace's native StaticCache mechanism.
    
    The model is loaded with use_cache=True and we pass a StaticCache
    that pre-allocates the KV buffers. We then register those buffers
    so they become Core ML states.
    
    This approach avoids custom attention classes entirely — it uses
    the standard HuggingFace code path.
    """
    
    def __init__(
        self,
        model_path: str,
        max_context_size: int = 2048,
    ) -> None:
        super().__init__()
        
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=DTYPE,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.config.use_cache = True
        
        # Create static cache
        self.static_cache = StaticCache(
            config=self.model.config,
            max_cache_len=max_context_size,
            batch_size=1,
        )
        
        # Force eager initialization of all layers
        # StaticCache uses lazy init — we need tensors now for register_buffer
        dummy_k = torch.zeros(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        dummy_v = torch.zeros(1, NUM_KV_HEADS, 1, HEAD_DIM, dtype=DTYPE)
        for layer_idx in range(NUM_LAYERS):
            self.static_cache.update(dummy_k, dummy_v, layer_idx)
        self.static_cache.reset()
        
        # Register each layer's cache as a buffer
        for layer_idx in range(NUM_LAYERS):
            layer = self.static_cache._cache_layers[layer_idx]
            self.register_buffer(
                f"key_cache_{layer_idx}",
                layer.key_cache,
            )
            self.register_buffer(
                f"value_cache_{layer_idx}",
                layer.value_cache,
            )
    
    @torch.no_grad()
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.model(
            input_ids,
            past_key_values=self.static_cache,
            use_cache=True,
        ).logits
```

### 6.3 Attempt export with StaticCache

```python
print("Loading StaticCacheQwen3Wrapper...")
static_wrapper = StaticCacheQwen3Wrapper(
    MODEL_ID,
    max_context_size=MAX_CONTEXT_SIZE,
)
static_wrapper.eval()

print(f"Registered buffers: {len(list(static_wrapper.named_buffers()))}")

# Try torch.export
example = (torch.zeros((1, 32), dtype=torch.long),)

print("Attempting torch.export with StaticCache...")
try:
    from torch.export import export, Dim
    seq_dim_d = Dim("seq_len", min=1, max=MAX_CONTEXT_SIZE)
    
    exported_d = export(
        static_wrapper,
        example,
        dynamic_shapes={"input_ids": {1: seq_dim_d}},
    )
    exported_d = exported_d.run_decompositions({})
    print("✅ torch.export with StaticCache succeeded")
    
except Exception as e:
    print(f"❌ torch.export failed: {e}")
    print("StaticCache may use dynamic control flow incompatible with export.")
    print("Trying torch.jit.trace as fallback...")
    
    try:
        trace_input = torch.zeros((1, 2), dtype=torch.int32)
        traced_d = torch.jit.trace(static_wrapper, [trace_input])
        print("✅ torch.jit.trace with StaticCache succeeded")
    except Exception as e2:
        print(f"❌ torch.jit.trace also failed: {e2}")
        traced_d = None
        exported_d = None
```

### 6.4 Convert to Core ML with per-layer states

```python
"""
If export or trace succeeded, convert to Core ML.
Per-layer state naming: key_cache_0, value_cache_0, ..., key_cache_27, value_cache_27
Total: 56 state tensors.
"""

per_layer_shape = (1, NUM_KV_HEADS, MAX_CONTEXT_SIZE, HEAD_DIM)

states_d = []
for layer_idx in range(NUM_LAYERS):
    states_d.append(
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=per_layer_shape,
                dtype=np.float16,
            ),
            name=f"key_cache_{layer_idx}",
        )
    )
    states_d.append(
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=per_layer_shape,
                dtype=np.float16,
            ),
            name=f"value_cache_{layer_idx}",
        )
    )

print(f"Total state tensors: {len(states_d)} (expected {NUM_LAYERS * 2})")

# Convert whichever succeeded
model_to_convert = None
if 'exported_d' in dir() and exported_d is not None:
    model_to_convert = exported_d
    print("Converting from torch.export...")
elif 'traced_d' in dir() and traced_d is not None:
    model_to_convert = traced_d
    print("Converting from torch.jit.trace...")

if model_to_convert is not None:
    seq_dim_d = ct.RangeDim(lower_bound=1, upper_bound=MAX_CONTEXT_SIZE, default=1)
    
    try:
        mlmodel_d = ct.convert(
            model_to_convert,
            inputs=[ct.TensorType(shape=(1, seq_dim_d), dtype=np.int32, name="inputIds")],
            outputs=[ct.TensorType(dtype=np.float16, name="logits")],
            states=states_d,
            minimum_deployment_target=ct.target.iOS18,
            skip_model_load=True,
        )
        mlmodel_d.save("models/Qwen3_1_7B_staticcache_fp16.mlpackage")
        print("✅ Core ML conversion complete (Approach D)")
    except Exception as e:
        print(f"❌ Core ML conversion failed: {e}")
else:
    print("⚠️ No model available for Core ML conversion (Approach D)")

del static_wrapper
```

---

## 7. Quantization (Shared)

> Applies to any of the above approaches after Core ML conversion.
> Use the same quantization pipeline from the existing notebook.

### 7.1 Int4 weight quantization (recommended)

```python
"""
Apply int4 block-wise symmetric weight quantization.
This is the same quantization used in the existing Qwen3 pipeline.
Reduces model size ~4x with minimal quality loss.
"""

def quantize_int4(mlmodel, output_path: str):
    op_config = ct.optimize.coreml.OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype="int4",
        granularity="per_block",
        block_size=32,
    )
    config = ct.optimize.coreml.OptimizationConfig(global_config=op_config)
    
    quantized = ct.optimize.coreml.linear_quantize_weights(mlmodel, config=config)
    quantized.save(output_path)
    print(f"✅ Int4 quantized model saved to {output_path}")
    return quantized

# Example usage (for whichever approach succeeded):
# quantize_int4(mlmodel_a, "models/Qwen3_1_7B_stateful_trace_w4.mlpackage")
# quantize_int4(mlmodel_c, "models/Qwen3_1_7B_explicit_kv_w4.mlpackage")
```

### 7.2 Int8 weight quantization (alternative)

```python
def quantize_int8(mlmodel, output_path: str):
    config = ct.optimize.coreml.OptimizationConfig(
        global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear",
            dtype="int8",
        )
    )
    
    quantized = ct.optimize.coreml.linear_quantize_weights(
        mlmodel,
        config=config,
        joint_compression=True,
    )
    quantized.save(output_path)
    print(f"✅ Int8 quantized model saved to {output_path}")
    return quantized
```

### 7.3 Mixed precision W4A8 (advanced)

```python
"""
Weight int4 + Activation int8 quantization.
Note: Activation quantization is experimental in coremltools 8.0 and may
fail on some compute units. Test CPU_ONLY first.
"""

def quantize_w4a8(mlmodel, calibration_samples, output_path: str):
    from coremltools.optimize import coreml as cto_coreml
    
    # Step 1: Activation int8 quantization
    activation_config = cto_coreml.OptimizationConfig(
        global_config=None,
        op_type_configs={
            "linear": cto_coreml.experimental.OpActivationLinearQuantizerConfig(
                mode="linear_symmetric"
            ),
            "matmul": cto_coreml.experimental.OpActivationLinearQuantizerConfig(
                mode="linear_symmetric"
            ),
        }
    )
    
    a8_model = cto_coreml.experimental.linear_quantize_activations(
        mlmodel,
        activation_config,
        calibration_samples,
    )
    
    # Step 2: Weight int4 quantization
    weight_config = cto_coreml.OptimizationConfig(
        global_config=cto_coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=32,
        ),
        op_type_configs={
            "layer_norm": None,
            "softmax": None,
            "rms_norm": None,
        },
    )
    
    w4a8_model = cto_coreml.linear_quantize_weights(
        a8_model,
        config=weight_config,
        joint_compression=True,
    )
    
    w4a8_model.save(output_path)
    print(f"✅ W4A8 model saved to {output_path}")
    return w4a8_model
```

---

## 8. Verification & Benchmarking

### 8.1 Numerical verification (stateful approach)

```python
"""
Compare stateful Core ML model output vs PyTorch reference.
IMPORTANT: State accumulates across calls — must reset between tests.
"""

def verify_stateful_model(mlmodel_path: str, tokenizer, ref_logits: np.ndarray):
    mlmodel = ct.models.MLModel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    
    # Create fresh state
    state = mlmodel.make_state()
    
    # Run prefill with test tokens
    test_prompt = "Hello, how are you"
    tokens = tokenizer(test_prompt, return_tensors="np")["input_ids"].astype(np.int32)
    seq_len = tokens.shape[1]
    
    mask = build_causal_mask(seq_len, seq_len)
    
    result = mlmodel.predict(
        {"inputIds": tokens, "causalMask": mask},
        state,
    )
    
    coreml_logits = result["logits"][0, -1, :].astype(np.float32)
    
    # Compare top-K predictions
    ref_top5 = np.argsort(ref_logits)[-5:][::-1]
    cml_top5 = np.argsort(coreml_logits)[-5:][::-1]
    
    print(f"PyTorch top-5: {ref_top5}")
    print(f"CoreML  top-5: {cml_top5}")
    print(f"Top-1 match: {ref_top5[0] == cml_top5[0]}")
    
    max_diff = np.max(np.abs(ref_logits - coreml_logits))
    print(f"Max logit difference: {max_diff:.4f}")
    
    return max_diff < 1.0  # FP16 + int4 can have notable differences
```

### 8.2 Stateful decode loop verification

```python
def verify_stateful_decode(mlmodel_path: str, tokenizer, num_steps: int = 10):
    """
    Verify the stateful decode loop produces coherent text.
    """
    mlmodel = ct.models.MLModel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    state = mlmodel.make_state()
    
    prompt = "The capital of France is"
    tokens = tokenizer(prompt, return_tensors="np")["input_ids"].astype(np.int32)
    seq_len = tokens.shape[1]
    
    generated = list(tokens[0])
    
    # Prefill
    mask = build_causal_mask(seq_len, seq_len)
    result = mlmodel.predict({"inputIds": tokens, "causalMask": mask}, state)
    next_token = int(np.argmax(result["logits"][0, -1, :]))
    generated.append(next_token)
    
    # Decode
    for step in range(num_steps - 1):
        total_len = seq_len + step + 1
        if total_len >= MAX_CONTEXT_SIZE:
            print(f"Reached max context at step {step}")
            break
        
        token_input = np.array([[next_token]], dtype=np.int32)
        mask = build_causal_mask(1, total_len + 1)
        
        result = mlmodel.predict({"inputIds": token_input, "causalMask": mask}, state)
        next_token = int(np.argmax(result["logits"][0, -1, :]))
        generated.append(next_token)
        
        if next_token == tokenizer.eos_token_id:
            break
    
    text = tokenizer.decode(generated)
    print(f"Generated: {text}")
    return text
```

### 8.3 State reset verification

```python
def verify_state_reset(mlmodel_path: str, tokenizer):
    """
    Verify that make_state() produces independent state objects,
    and that results are deterministic when starting from fresh state.
    """
    mlmodel = ct.models.MLModel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    
    tokens = np.array([[1, 2, 3]], dtype=np.int32)
    mask = build_causal_mask(3, 3)
    
    # Run with state 1
    state1 = mlmodel.make_state()
    result1 = mlmodel.predict({"inputIds": tokens, "causalMask": mask}, state1)
    logits1 = result1["logits"][0, -1, :]
    
    # Run with state 2 (independent)
    state2 = mlmodel.make_state()
    result2 = mlmodel.predict({"inputIds": tokens, "causalMask": mask}, state2)
    logits2 = result2["logits"][0, -1, :]
    
    diff = np.max(np.abs(logits1.astype(np.float32) - logits2.astype(np.float32)))
    print(f"State independence check — max diff: {diff}")
    assert diff < 1e-6, "States are not independent!"
    
    # Run state 1 again — should give DIFFERENT results (cache is populated)
    result1b = mlmodel.predict(
        {"inputIds": np.array([[4]], dtype=np.int32), "causalMask": build_causal_mask(1, 4)},
        state1,
    )
    logits1b = result1b["logits"][0, -1, :]
    
    # state2 with same token should give different results (different history)
    result2b = mlmodel.predict(
        {"inputIds": np.array([[4]], dtype=np.int32), "causalMask": build_causal_mask(1, 4)},
        state2,
    )
    logits2b = result2b["logits"][0, -1, :]
    
    # They should be the same since both states saw [1,2,3] then [4]
    diff2 = np.max(np.abs(logits1b.astype(np.float32) - logits2b.astype(np.float32)))
    print(f"Same-history check — max diff: {diff2}")
    assert diff2 < 1e-6, "Same history should produce same output!"
    
    print("✅ State reset verification passed")
```

### 8.4 Performance benchmark

```python
import time

def benchmark_model(mlmodel_path: str, num_tokens: int = 100, use_state: bool = True):
    """
    Benchmark token generation throughput.
    """
    mlmodel = ct.models.MLModel(mlmodel_path, compute_units=ct.ComputeUnit.CPU_AND_GPU)
    
    state = mlmodel.make_state() if use_state else None
    
    # Prefill
    tokens = np.random.randint(0, VOCAB_SIZE, (1, 5), dtype=np.int32)
    mask = build_causal_mask(5, 5)
    
    predict_kwargs = {"inputIds": tokens, "causalMask": mask}
    
    if use_state:
        mlmodel.predict(predict_kwargs, state)
    else:
        predict_kwargs["keyCacheIn"] = np.zeros(kv_shape, dtype=np.float16)
        predict_kwargs["valueCacheIn"] = np.zeros(kv_shape, dtype=np.float16)
        result = mlmodel.predict(predict_kwargs)
    
    # Decode benchmark
    past_len = 5
    start = time.perf_counter()
    
    for step in range(num_tokens):
        token = np.array([[step % VOCAB_SIZE]], dtype=np.int32)
        mask = build_causal_mask(1, past_len + 1)
        
        predict_kwargs = {"inputIds": token, "causalMask": mask}
        
        if use_state:
            mlmodel.predict(predict_kwargs, state)
        else:
            predict_kwargs["keyCacheIn"] = result.get("keyCacheOut", np.zeros(kv_shape, dtype=np.float16))
            predict_kwargs["valueCacheIn"] = result.get("valueCacheOut", np.zeros(kv_shape, dtype=np.float16))
            result = mlmodel.predict(predict_kwargs)
        
        past_len += 1
    
    elapsed = time.perf_counter() - start
    tok_per_sec = num_tokens / elapsed
    ms_per_tok = (elapsed / num_tokens) * 1000
    
    print(f"Generated {num_tokens} tokens in {elapsed:.2f}s")
    print(f"  {tok_per_sec:.1f} tokens/sec")
    print(f"  {ms_per_tok:.1f} ms/token")
    
    return tok_per_sec
```

---

## 9. Comparison Matrix

| Aspect | A: Trace+State | B: Export+State | C: Explicit I/O | D: StaticCache |
|--------|---------------|-----------------|------------------|----------------|
| **Export method** | `torch.jit.trace` | `torch.export` | `torch.jit.trace` | `torch.export` or `trace` |
| **Core ML States** | ✅ Yes (MLState) | ✅ Yes (MLState) | ❌ No | ✅ Yes (MLState) |
| **Custom attention** | ✅ Required | ✅ Required | ✅ Required | ❌ Uses HF native |
| **Proven reference** | ✅ Mistral 7B (Apple) | ⚠️ Experimental | ✅ Well-understood | ⚠️ Experimental |
| **Performance** | 🟢 ~13x faster | 🟢 ~13x faster | 🔴 Baseline (slow) | 🟢 ~13x faster |
| **Memory (decode)** | 🟢 Cache on-GPU | 🟢 Cache on-GPU | 🔴 Copy per step | 🟢 Cache on-GPU |
| **Implementation risk** | 🟡 Medium | 🟡 Medium-High | 🟢 Low | 🟡 Medium-High |
| **State tensors** | 2 (combined) | 2 (combined) | 0 (I/O pairs) | 56 (per-layer) |
| **iOS 18 required** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Dynamic context** | `ct.RangeDim` | `Dim` + `RangeDim` | `ct.RangeDim` | `ct.RangeDim` |

### Recommended Priority

1. **Approach A** (Trace + State): Start here. Proven pattern from Apple/HuggingFace Mistral 7B.
2. **Approach C** (Explicit I/O): Fallback if stateful approach has issues. Functional but slower.
3. **Approach B** (Export + State): Try if `torch.export` handles buffer mutations correctly.
4. **Approach D** (StaticCache): Explore if HuggingFace's native cache aligns with Core ML states.

### Key Considerations

- **KV cache shape**: Approaches A/B/C use combined shape `(28, 1, 4, 2048, 128)` — 2 state tensors. Approach D uses per-layer shape `(1, 4, 2048, 128)` — 56 state tensors. Combined is simpler but per-layer may be more flexible.
- **Qwen3 QK Norm**: All custom attention approaches must preserve the `q_norm` and `k_norm` RMSNorm layers. Missing these will produce incorrect attention patterns.
- **Sliding window**: Qwen3 config may specify some layers as sliding window attention. The custom attention class should respect `self.sliding_window` if present. For the 1.7B model, verify `config.layer_types` to see if any layers use sliding window.
- **Apple's RangeDim for states**: state tensor shapes are fixed at conversion time (not dynamic). The full 2048 context is pre-allocated. However, the causal mask controls which positions are actually attended to, making this memory-efficient in practice — unused positions are never read.

---

## Appendix: File Listing

After running the approaches, the expected output files:

```
models/
├── Qwen3_1_7B_stateful_trace_fp16.mlpackage     # Approach A (FP16)
├── Qwen3_1_7B_stateful_trace_w4.mlpackage        # Approach A (Int4)
├── Qwen3_1_7B_stateful_export_fp16.mlpackage     # Approach B (FP16)
├── Qwen3_1_7B_explicit_kv_fp16.mlpackage         # Approach C (FP16)
├── Qwen3_1_7B_explicit_kv_w4.mlpackage           # Approach C (Int4)
├── Qwen3_1_7B_staticcache_fp16.mlpackage         # Approach D (FP16)
```
