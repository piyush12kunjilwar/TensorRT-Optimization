# TensorRT Optimization 🚀

> LLaMA-style transformer optimized with TensorRT FP16 —
> achieving 12.82x speedup over PyTorch FP32 baseline

## Hardware
- **GPU:** NVIDIA L4 (23.7GB, CUDA Capability 8.9)
- **CUDA:** 12.8 | **TensorRT:** 10.14 | **PyTorch:** 2.10

---

## Results

| Method | Mean Latency | P95 | vs Baseline | Speedup |
|--------|-------------|-----|-------------|---------|
| PyTorch FP32 | 3.429ms | 3.514ms | 1.00x | baseline |
| PyTorch FP16 | 3.233ms | 3.310ms | 0.94x | 1.06x |
| ONNX Runtime | 6.138ms | 6.484ms | 1.79x | 0.56x |
| **TensorRT FP16** | **0.267ms** | **0.276ms** | **0.08x** | **12.82x** |

**Model:** LLaMA-style Transformer (10.8M params, 4 layers)
**Config:** Batch=1, SeqLen=128

---

## What TensorRT Does Internally

### 1. Layer Fusion
```
Before TRT:  RMSNorm → Linear → SwiGLU → Linear
             = 4 separate kernel launches
             = 4 HBM round trips

After TRT:   RMSNorm+Linear+SwiGLU+Linear
             = 1 fused kernel
             = 1 HBM round trip
```

### 2. Kernel Auto-tuning
- Tests thousands of CUDA kernel implementations
- Selects fastest for YOUR specific GPU
- Accounts for batch size, sequence length, memory
- Why engine build takes 2-5 minutes
- Result: optimal kernel for every single layer

### 3. FP16 Tensor Cores
- L4 GPU has dedicated FP16 matrix multiply hardware
- TRT routes all matmuls to tensor cores automatically
- 10x+ faster than FP32 CUDA cores for same operation

### 4. Memory Optimization
- Tensor reuse across layers
- In-place operations where safe
- Optimal memory layout per operation
- Eliminates redundant allocations

### 5. CUDA Graph Capture
- Entire forward pass captured as one graph
- Single GPU kernel launch per inference
- Eliminates CPU overhead between layers

---

## Optimization Pipeline
```
PyTorch Model
     ↓
ONNX Export (torch.onnx, opset 17, constant folding)
     ↓
TensorRT Engine Build
  - Layer fusion
  - FP16 precision
  - Kernel selection
  - Memory optimization
     ↓
Serialized .trt Engine
     ↓
Production Inference: 0.267ms
```

---

## LLaMA Architecture Used
| Component | Choice | Why |
|-----------|--------|-----|
| Normalization | RMSNorm | 15% faster than LayerNorm |
| Activation | SwiGLU | Better than ReLU/GELU |
| Attention | Scaled dot-product | Flash Attention compatible |
| Bias | None | Cleaner gradients |

---

## Cross-Module Journey

| Module | Optimization | Result |
|--------|-------------|--------|
| Module 01 | INT8 Quantization | 1.94x speedup |
| Module 02 | Triton Kernels | 7.54x speedup |
| Module 03 | FSDP + Grad Accum | 50.4% memory savings |
| **Module 04** | **TensorRT FP16** | **12.82x speedup** |

**Production stack:**
- Train with: FSDP + AMP + Gradient Accumulation
- Deploy with: TensorRT FP16 engine
- Result: World-class training AND inference

---

## Interview Answer
*"I optimized a LLaMA-style transformer for production
inference using TensorRT. Starting from PyTorch FP32 at
3.429ms, I exported to ONNX then built a TensorRT FP16
engine. TensorRT performed layer fusion — combining 50+
kernel launches into fused operations — selected optimal
CUDA kernels for the L4 GPU through auto-tuning, and
routed matrix multiplications to FP16 tensor cores.
Final result: 0.267ms — 12.82x faster than baseline."*

---

## Tech Stack
```
TensorRT 10.14 · ONNX 1.20 · ONNXRuntime 1.24
PyTorch 2.10 · CUDA 12.8 · NVIDIA L4 GPU
Python 3.12 · Google Colab Pro
```

---

## Part of ML Systems Optimization Suite
- ✅ Module 1 — Inference Optimization (ONNX + Quantization)
- ✅ Module 2 — CUDA Kernel Optimization (Triton + Flash Attention)
- ✅ Module 3 — Distributed Training (FSDP + NCCL)
- ✅ Module 4 — TensorRT Optimization (this repo)
- 🔜 Module 5 — Agentic AI Systems

---

## Author
**Piyush Kunjilwar**
MS Information Systems — Northeastern University (May 2026)
[LinkedIn](https://linkedin.com/in/piyush-kunjilwar) ·
[GitHub](https://github.com/piyush12kunjilwar) ·
[Portfolio](https://piyush12kunjilwar.github.io)# TensorRT-Optimization
