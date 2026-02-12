# OBJECTIVE

A w4a8 quantized version of Qwen3 1.7B model in the CoreML mlmodel format

# TRIED SO FAR 

## Load from hugging face -> Try quantization with coremltools.optimize.torch.quantization import LinearQuantizer, LinearQuantizerConfig, ModuleLinearQuantizerConfig -> Export to CoreML mlmodel

- This failed at the quantization step giving: ValueError: code: co_varnames is too small

## Load from hugging face -> Export to Core ML at FP 16 -> Quantize the Core ML model to w4a8

- This dailed at the export step giving: ValueError: In op, of type sub, named sub_1, the named input `y` must have the same data type as the named input `x`. However, y has dtype fp16 whereas x has dtype fp32.

In both cases, I have tried using a wrapper for Qwen3 to make it compatible with the expectations of CoreML. This wrapper has caused other issues such as not fully exporting the model graph

## Export through Executorch

- This failed for both Core ML and MPS because the Executorch api did not support conversion for these backends to int4 quantization

# REFERENCES

- All chats in this project
- qwen3_coreml_w4a8.ipynb file - which is a Jupyter notebook [MAIN CODE which needs to be analyzed]
- https://developer.apple.com/documentation/coreml
- https://apple.github.io/coremltools/index.html
- https://apple.github.io/coremltools/docs-guides/source/opt-overview.html
- https://github.com/pytorch/pytorch
- https://github.com/pytorch/ao
- https://huggingface.co
- https://arxiv.org/abs/2505.09388

# WHAT I WANT YOU TO DO

1. Check on Hugging Face if there are Qwen3 1.7B models quantized to w4a8 in the Core ML format
2. Analyze the code in qwen3_coreml_w4a8.ipynb and identify the cause of errors and suggest fixes
3. Suggest verified approaches grounded in official documentation of going about this export
4. Take into account Qwen3 specific nuances which need to be considered during export

I want to a fool proof tried and tested verified approach to get quantized Qwen3 1.7B on CoreML
DO NOT generate custom code unless absolutely required. USE existing functions from the published APIs





