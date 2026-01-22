# Small Language Model Training & Inference Engine
## Enterprise Architecture Document

**Date:** January 2026
**Team Size:** 2 Developers
**Cloud Platform:** Microsoft Azure

---

## Table of Contents
1. [Enterprise Small Language Models](#1-enterprise-small-language-models)
2. [Inference Optimization Techniques](#2-inference-optimization-techniques)
3. [Training Optimization Techniques](#3-training-optimization-techniques)
4. [GPU Memory Requirements by Context Length](#4-gpu-memory-requirements-by-context-length)
5. [Azure Infrastructure & Cost Estimates](#5-azure-infrastructure--cost-estimates)
6. [Recommended Architecture](#6-recommended-architecture)

---

## 1. Enterprise Small Language Models

### 1.1 Recommended Models Comparison

| Model | Parameters | Context Window | License | Best For |
|-------|------------|----------------|---------|----------|
| **Qwen3-8B** | 8B | 128K (extendable to 1M) | Apache 2.0 | General enterprise tasks, multilingual (119 languages) |
| **Qwen3-30B-A3B** (MoE) | 30B total / 3B active | 128K | Apache 2.0 | High performance with low inference cost |
| **Phi-4** | 16B | 16K | MIT | Reasoning, math, code generation |
| **Phi-4-mini** | 3.8B | 128K | MIT | Resource-constrained deployments |
| **Llama 3.1-8B** | 8B | 128K | Llama 3 License | General purpose, large ecosystem |
| **Mistral 7B** | 7.3B | 32K | Apache 2.0 | Balanced efficiency, runs on 16GB RAM |
| **Mistral Small 3.1** | 24B | 128K | Apache 2.0 | High throughput (~150 tokens/sec) |
| **Gemma 3-12B** | 12B | 128K | Gemma License | Multimodal (vision + audio), on-device |

### 1.2 Detailed Model Analysis

#### **Qwen3 (Alibaba) - TOP RECOMMENDATION**

**Rationale for Selection:**
- **50% density improvement**: Qwen3-8B performs equivalent to Qwen2.5-14B
- **MoE efficiency**: Qwen3-30B-A3B delivers flagship performance with only 3B active parameters (10x parameter efficiency)
- **119 language support** - best multilingual coverage
- **Unified thinking modes**: Dynamic task-adaptive computation for reasoning vs. rapid response
- **Apache 2.0 license** - full commercial use without restrictions
- **Cost efficiency**: Open-source models deliver 89-94% of proprietary capabilities at 15-25% of costs

**Available Sizes:**
- Dense: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- MoE: 30B-A3B (recommended), 235B-A22B

**Framework Support:** ✅ vLLM, ✅ TensorRT-LLM, ✅ HuggingFace, ✅ SGLang

---

#### **Phi-4 (Microsoft)**

**Rationale for Selection:**
- **Data quality focus**: Trained on synthetic data, filtered content, academic resources
- **Strong reasoning**: Excels at math, code, and logical tasks
- **Azure-native**: Optimized for Azure deployment with first-party support
- **Small footprint**: Phi-4-mini (3.8B) matches Llama-3.1-8B performance

**Available Sizes:**
- Phi-4: 16B parameters
- Phi-4-mini: 3.8B parameters
- Phi-3.5: 3.8B (mini), 7B (small), 14B (medium), 42B (MoE)

**Framework Support:** ✅ vLLM, ✅ TensorRT-LLM, ✅ ONNX, ✅ Azure ML

---

#### **Llama 3.1/4 (Meta)**

**Rationale for Selection:**
- **Largest ecosystem**: Most community support, tutorials, and tooling
- **Proven production use**: Battle-tested in enterprise deployments
- **Llama 4 MoE**: Scout and Maverick variants with mixture-of-experts architecture

**Available Sizes:**
- Llama 3.1: 8B, 70B, 405B
- Llama 4: Scout, Maverick (MoE architecture)

**Framework Support:** ✅ vLLM, ✅ TensorRT-LLM, ✅ llama.cpp, ✅ All major frameworks

---

#### **Mistral 7B / Small 3.1**

**Rationale for Selection:**
- **Best efficiency**: Runs smoothly on consumer hardware (16GB RAM)
- **Mistral Small 3.1**: ~150 tokens/sec streaming, single consumer GPU capable
- **Production proven**: Strong community praise for reliability

**Framework Support:** ✅ vLLM, ✅ TensorRT-LLM, ✅ Ollama, ✅ llama.cpp

---

#### **Gemma 3 (Google)**

**Rationale for Selection:**
- **Multimodal native**: Vision encoder + audio encoder built-in (Gemma 3n)
- **On-device optimized**: Designed for edge deployment
- **Google pedigree**: Same research foundation as Gemini

**Available Sizes:** 1B, 4B, 12B, 27B

**Framework Support:** ✅ vLLM, ✅ TensorRT-LLM, ✅ MediaPipe, ✅ TensorFlow Lite

---

### 1.3 Model Selection Decision Matrix

| Use Case | Recommended Model | Rationale |
|----------|-------------------|-----------|
| **General Enterprise Chat** | Qwen3-8B | Best multilingual, Apache 2.0, strong performance |
| **Cost-Optimized High Performance** | Qwen3-30B-A3B (MoE) | 10x parameter efficiency, flagship quality |
| **Code Generation** | Phi-4 or Qwen3-Coder | Purpose-built for code tasks |
| **Resource Constrained** | Phi-4-mini (3.8B) or Mistral 7B | Excellent performance/size ratio |
| **Multimodal (Vision/Audio)** | Gemma 3n or Qwen3-VL | Native multimodal support |
| **Maximum Ecosystem Support** | Llama 3.1-8B | Largest community, most tooling |

---

## 2. Inference Optimization Techniques

### 2.1 Optimization Techniques Summary

| Technique | Speedup | Memory Savings | Framework | Custom Dev Needed |
|-----------|---------|----------------|-----------|-------------------|
| **Continuous Batching** | 2-4x throughput | Moderate | vLLM, TensorRT-LLM | ❌ No |
| **PagedAttention** | 2-4x throughput | 50-70% KV cache | vLLM | ❌ No |
| **Quantization (FP8/INT4)** | 1.5-3x | 50-75% | TensorRT-LLM, vLLM | ❌ No |
| **Speculative Decoding** | 2-3x latency | Minimal | vLLM, TensorRT-LLM | ❌ No |
| **EAGLE-3** | 2-6.5x latency | Minimal | vLLM, SGLang | ⚠️ Training required |
| **KV Cache Optimization** | 1.5-2x | 40-60% | LMDeploy, vLLM | ❌ No |
| **Tensor Parallelism** | Linear scaling | Distributed | All major frameworks | ❌ No |

---

### 2.2 Detailed Technique Analysis

#### **2.2.1 Continuous Batching**

**How it Works:**
- Evicts completed sequences and inserts new ones dynamically
- Eliminates head-of-line blocking where short requests wait for long ones
- Maximizes GPU utilization by keeping batch sizes optimal

**Performance:** 2-4x throughput improvement over static batching

**Framework Implementation:**
| Framework | Feature Name | Status |
|-----------|--------------|--------|
| vLLM | Continuous Batching | ✅ Native |
| TensorRT-LLM | Inflight Batching | ✅ Native |
| LMDeploy | Persistent Batching | ✅ Native |
| HF TGI | Dynamic Batching | ✅ Native |

**Custom Development:** ❌ Not required - use existing frameworks

---

#### **2.2.2 PagedAttention**

**How it Works:**
- Treats KV cache like virtual memory with non-contiguous pages
- Enables KV cache reuse and sharing across requests
- Eliminates memory fragmentation that causes OOM errors

**Performance:** 2-4x throughput at same latency, especially for long sequences

**Framework Implementation:**
| Framework | Status |
|-----------|--------|
| vLLM | ✅ Native (invented here) |
| TensorRT-LLM | ✅ Paged KV Caching |
| SGLang | ✅ Native |
| LMDeploy | ✅ Blocked KV |

**Custom Development:** ❌ Not required

---

#### **2.2.3 Quantization**

**Quantization Formats:**

| Format | Memory Reduction | Speed Improvement | Quality Loss | Best For |
|--------|------------------|-------------------|--------------|----------|
| **FP16/BF16** | Baseline | Baseline | None | Training, high-quality inference |
| **FP8** | 50% | 1.5-2x | Minimal | Production inference on Hopper+ GPUs |
| **INT8** | 50% | 1.5-2x | Small | General inference |
| **INT4/FP4** | 75% | 2-3x | Moderate | Edge deployment, cost optimization |
| **NF4 (QLoRA)** | 75% | 2x | Small | Fine-tuning memory reduction |

**Framework Implementation:**
| Framework | Supported Formats | Tooling |
|-----------|-------------------|---------|
| TensorRT-LLM | FP8, FP4, INT8, INT4 | NVIDIA Model Optimizer |
| vLLM | FP8, INT8, INT4, AWQ, GPTQ | Native + AutoAWQ |
| llama.cpp | Q4_K_M, Q5_K_M, Q8_0 | GGUF format |
| ONNX Runtime | INT8, INT4 | ONNX Quantization |

**Custom Development:** ❌ Not required - use NVIDIA Model Optimizer or AutoAWQ

---

#### **2.2.4 Speculative Decoding**

**How it Works:**
1. **Draft Phase**: Small/fast model generates K candidate tokens
2. **Verify Phase**: Target model verifies all K tokens in parallel (single forward pass)
3. **Accept/Reject**: Rejection sampling determines which tokens to keep
4. **Guarantee**: Output is mathematically identical to standard decoding

**Performance:**
- 2-3x latency reduction at acceptance rate ≥60%
- Works best when draft model achieves high acceptance rate
- Memory-bound nature of LLM inference leaves GPU compute underutilized

**Framework Implementation:**
| Framework | Draft Model Support | Status |
|-----------|---------------------|--------|
| vLLM | External draft model, EAGLE | ✅ Production ready |
| TensorRT-LLM | External draft model | ✅ Production ready |
| SGLang | EAGLE-3, external | ✅ Production ready |

**Draft Model Selection:**
- Use smaller model from same family (e.g., Qwen3-0.6B for Qwen3-8B)
- Latency of draft model more important than capability
- Target acceptance rate: ≥60% for meaningful speedup

**Custom Development:** ❌ Not required - frameworks support this natively

---

#### **2.2.5 EAGLE-3 (Advanced Speculative Decoding)**

**How it Works:**
- Attaches lightweight "draft head" (2-5% of target model size) to target model's hidden layers
- Abandons feature prediction for direct token prediction
- Uses multi-layer feature fusion via "training-time test"
- Generates tree of candidate tokens, not just linear sequence

**Performance:**
- **2-6.5x speedup** over standard autoregressive generation
- Better scaling with more training data than EAGLE-1/2
- vLLM reports up to 2.5x boost across diverse scenarios
- Llama 4 Maverick draft: 2.18x speedup on MT-Bench

**Framework Implementation:**
| Framework | EAGLE-3 Support | Status |
|-----------|-----------------|--------|
| vLLM | ✅ Native | Production ready |
| SGLang | ✅ Native + SpecForge training | Production ready |
| TensorRT-LLM | ✅ Supported | Production ready |

**Training Requirements:**
- Need to train EAGLE-3 draft head for your specific target model
- **SpecForge**: Training framework tightly integrated with SGLang
- Training data: General text corpus, scales well with more data

**Custom Development:** ⚠️ Partial - Need to train draft head, but frameworks and tools exist

---

#### **2.2.6 Inference Framework Comparison**

| Framework | Throughput | Latency (TTFT) | Best For | Ease of Use |
|-----------|------------|----------------|----------|-------------|
| **vLLM** | 120-160 req/sec | 50-80ms | General production, OpenAI-compatible API | ⭐⭐⭐⭐⭐ |
| **TensorRT-LLM** | 180-220 req/sec | 35-50ms | Maximum NVIDIA performance, latency-critical | ⭐⭐⭐ |
| **LMDeploy** | 1.8x vLLM | Excellent | Quantized models, 4-bit Llama family | ⭐⭐⭐⭐ |
| **SGLang** | High | Low | Complex prompting, structured generation | ⭐⭐⭐⭐ |
| **HF TGI** | 100-140 req/sec | 60-90ms | HuggingFace ecosystem, simplicity | ⭐⭐⭐⭐⭐ |

**Recommendation:**
- **Start with vLLM** for fastest time-to-production and OpenAI-compatible APIs
- **Migrate to TensorRT-LLM** if you need maximum throughput and are committed to NVIDIA
- **Use LMDeploy** for 4-bit quantized models with blocked KV optimization

---

### 2.3 Inference Optimization Implementation Roadmap

```
Phase 1 (Week 1-2): Basic Deployment
├── Deploy with vLLM
├── Enable continuous batching (default)
├── Enable PagedAttention (default)
└── Benchmark baseline performance

Phase 2 (Week 3-4): Quantization
├── Apply FP8 quantization (Hopper GPUs)
├── Or INT8/INT4 for older GPUs
├── Use NVIDIA Model Optimizer
└── Validate quality with evaluation suite

Phase 3 (Week 5-6): Speculative Decoding
├── Enable speculative decoding with draft model
├── Or train EAGLE-3 draft head
├── Tune acceptance rate targets
└── Benchmark latency improvements

Phase 4 (Ongoing): Production Optimization
├── Monitor and tune batch sizes
├── Implement request prioritization
├── Add autoscaling based on load
└── Consider TensorRT-LLM migration for max throughput
```

---

## 3. Training Optimization Techniques

### 3.1 Training Techniques Summary

| Technique | Memory Reduction | Training Speed | Quality Impact | Framework |
|-----------|------------------|----------------|----------------|-----------|
| **LoRA** | 90-95% | 2-3x faster | Minimal | HuggingFace PEFT |
| **QLoRA** | 95-97% | 2x faster | Minimal | HuggingFace PEFT, bitsandbytes |
| **DoRA** | 90-95% | 2-3x faster | Better than LoRA | HuggingFace PEFT |
| **Full Fine-tuning** | Baseline | Baseline | Best | PyTorch, DeepSpeed |
| **Gradient Checkpointing** | 60-70% | 20-30% slower | None | Native PyTorch |
| **Mixed Precision (BF16)** | 50% | 1.5-2x | Minimal | Native PyTorch |
| **DeepSpeed ZeRO** | 80-95% | Enables larger models | None | DeepSpeed |
| **FSDP** | 80-90% | Enables larger models | None | PyTorch native |

---

### 3.2 Detailed Training Techniques

#### **3.2.1 LoRA (Low-Rank Adaptation)**

**How it Works:**
- Instead of updating full weight matrix W, decomposes update into two smaller matrices: W + B·A
- B and A are low-rank matrices with far fewer parameters
- Only 0.5-5% of parameters are trainable

**Where to Apply LoRA:**
- **Original recommendation**: Attention matrices only
- **Current best practice**: Apply to ALL layers including MLP
- Research shows MLP-only LoRA outperforms attention-only LoRA

**Configuration:**
```python
# Recommended LoRA config
lora_config = LoraConfig(
    r=16,                    # Rank (8-64 typical)
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Apply to all linear layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Framework Implementation:**
| Framework | Status | Notes |
|-----------|--------|-------|
| HuggingFace PEFT | ✅ Native | Recommended, best documentation |
| Axolotl | ✅ Native | Easy config-based training |
| LLaMA-Factory | ✅ Native | Web UI available |
| Unsloth | ✅ Native | 2x faster training |

**Custom Development:** ❌ Not required

---

#### **3.2.2 QLoRA (Quantized LoRA)**

**How it Works:**
1. Quantize base model to 4-bit NormalFloat (NF4) format
2. Train LoRA adapters on top of quantized model
3. Double quantization: Quantize both weights AND quantization constants

**Memory Requirements:**

| Model Size | Full Fine-tune | LoRA | QLoRA |
|------------|----------------|------|-------|
| 7B | ~120 GB | ~30 GB | ~6-8 GB |
| 13B | ~200 GB | ~50 GB | ~12 GB |
| 70B | ~1.2 TB | ~140 GB | ~20-24 GB |

**Practical Example:**
- Fine-tune Llama-2-7B with QLoRA on a 16GB GPU (V100/T4)
- Fine-tune 70B model on single GPU with 20GB HBM

**Framework Implementation:**
| Framework | Status | Notes |
|-----------|--------|-------|
| HuggingFace PEFT + bitsandbytes | ✅ Native | Standard approach |
| Axolotl | ✅ Native | Config-based |
| Unsloth | ✅ Native | Optimized kernels, 2x faster |
| TRL | ✅ Native | For RLHF/DPO training |

**Custom Development:** ❌ Not required

---

#### **3.2.3 DoRA (Weight-Decomposed Low-Rank Adaptation)**

**How it Works:**
- Decomposes weight matrix into magnitude and direction components
- Fine-tunes each component separately
- Better gradient flow than standard LoRA

**Performance:** Consistently outperforms LoRA with same parameter budget

**Framework Implementation:**
| Framework | Status |
|-----------|--------|
| HuggingFace PEFT | ✅ Native |

**Custom Development:** ❌ Not required

---

#### **3.2.4 Distributed Training (DeepSpeed / FSDP)**

**DeepSpeed ZeRO Stages:**

| Stage | Memory Savings | Communication | Use Case |
|-------|----------------|---------------|----------|
| ZeRO-1 | Optimizer states partitioned | Low | Multi-GPU, moderate savings |
| ZeRO-2 | + Gradients partitioned | Medium | Larger models |
| ZeRO-3 | + Parameters partitioned | High | Maximum memory efficiency |
| ZeRO-Offload | CPU/NVMe offloading | Variable | GPU memory constrained |

**FSDP (Fully Sharded Data Parallel):**
- PyTorch native alternative to DeepSpeed
- Shards model parameters, gradients, and optimizer states
- Good integration with HuggingFace Trainer

**Framework Implementation:**
| Framework | DeepSpeed | FSDP |
|-----------|-----------|------|
| HuggingFace Trainer | ✅ Native | ✅ Native |
| PyTorch Lightning | ✅ Native | ✅ Native |
| Axolotl | ✅ Native | ✅ Native |

**Custom Development:** ❌ Not required

---

#### **3.2.5 Mixed Precision Training**

**Formats:**
- **FP32**: Full precision (baseline)
- **FP16**: Half precision, requires loss scaling
- **BF16**: Brain float 16, recommended for modern GPUs (A100, H100)
- **FP8**: Available on Hopper+ for training (experimental)

**Recommendation:** Use BF16 on A100/H100, FP16 with loss scaling on older GPUs

**Custom Development:** ❌ Not required - native in PyTorch

---

### 3.3 Training Framework Comparison

| Framework | Ease of Use | Features | Best For |
|-----------|-------------|----------|----------|
| **Axolotl** | ⭐⭐⭐⭐⭐ | Config-based, all PEFT methods | Quick experimentation |
| **Unsloth** | ⭐⭐⭐⭐⭐ | 2x faster, memory efficient | Speed-critical training |
| **HuggingFace TRL** | ⭐⭐⭐⭐ | RLHF, DPO, PPO | Alignment training |
| **LLaMA-Factory** | ⭐⭐⭐⭐⭐ | Web UI, 100+ models | Non-technical users |
| **torchtune** | ⭐⭐⭐ | PyTorch native, modular | Custom implementations |

---

### 3.4 Training Implementation Roadmap

```
Phase 1: Environment Setup
├── Install CUDA, PyTorch, HuggingFace stack
├── Set up Axolotl or Unsloth
├── Configure DeepSpeed/FSDP
└── Prepare dataset in appropriate format

Phase 2: Initial Fine-tuning
├── Start with QLoRA for memory efficiency
├── Apply to all layers (not just attention)
├── Use BF16 mixed precision
└── Implement gradient checkpointing

Phase 3: Scale Up
├── Increase LoRA rank if quality insufficient
├── Try DoRA for improved performance
├── Add more training data
└── Implement distributed training if needed

Phase 4: Alignment (Optional)
├── SFT (Supervised Fine-Tuning)
├── DPO (Direct Preference Optimization)
└── Evaluate with standardized benchmarks
```

---

## 4. GPU Memory Requirements by Context Length

### 4.1 Why Context Length Matters for Memory

Context length significantly impacts GPU memory requirements due to:

1. **Activations scale with sequence length** - even with gradient checkpointing
2. **KV Cache grows linearly** - `seq_length × layers × heads × head_dim`
3. **Attention computation** - O(n²) without Flash Attention, O(n) with Flash Attention

### 4.2 Memory Scaling Analysis for Qwen3-8B

| Component | 2K Context | 32K Context | Scaling Factor |
|-----------|------------|-------------|----------------|
| Base model (4-bit NF4) | ~5 GB | ~5 GB | Fixed |
| LoRA adapters (BF16) | ~200 MB | ~200 MB | Fixed |
| Optimizer states (AdamW) | ~500 MB | ~500 MB | Fixed |
| Gradients | ~200 MB | ~200 MB | Fixed |
| **Activations** | ~3 GB | ~40-50 GB | ~16x (linear) |
| **KV Cache** | ~0.5 GB | ~8 GB | 16x (linear) |
| CUDA overhead | ~1 GB | ~2 GB | Variable |
| **TOTAL** | **~10-12 GB** | **~55-65 GB** | ~5-6x |

### 4.3 GPU Requirements by Context Length (QLoRA Qwen3-8B)

| Context Length | Min GPU Memory | Recommended GPU | Batch Size | Azure Instance |
|----------------|----------------|-----------------|------------|----------------|
| **2K tokens** | 12 GB | T4 (16 GB) | 2-4 | NC8as T4 v3 |
| **4K tokens** | 16 GB | T4 (16 GB) | 1-2 | NC8as T4 v3 |
| **8K tokens** | 20-24 GB | A10 (24 GB) | 1-2 | NC8as T4 v3 (tight) |
| **16K tokens** | 32-40 GB | A100 40GB | 1 | NC24ads A100 v4 |
| **32K tokens** | 55-65 GB | A100 80GB | 1 | NC24ads A100 v4 |
| **64K+ tokens** | 80+ GB | A100 80GB or H100 | 1 | NC24ads A100 v4 / H100 |

### 4.4 Detailed 32K Context Configuration

**Required Optimizations (MANDATORY for 32K):**

```python
# 1. Flash Attention 2 - CRITICAL
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    attn_implementation="flash_attention_2",  # Reduces O(n²) to O(n)
    torch_dtype=torch.bfloat16
)

# 2. Gradient Checkpointing - CRITICAL
model.gradient_checkpointing_enable()

# 3. 4-bit Quantization for QLoRA
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Double quantization
)
```

**Recommended Training Configuration for 32K:**

```yaml
# Axolotl config for Qwen3-8B @ 32K context
model_type: AutoModelForCausalLM
base_model: Qwen/Qwen3-8B

sequence_len: 32768
sample_packing: true  # Pack multiple samples efficiently

# QLoRA settings
adapter: qlora
lora_r: 16
lora_alpha: 32
lora_target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj

# Memory optimizations
load_in_4bit: true
bf16: true
gradient_checkpointing: true
flash_attention: true

# Batch settings for 80GB GPU
micro_batch_size: 1
gradient_accumulation_steps: 16
# Effective batch size: 16
```

### 4.5 Alternative: Using Unsloth for 32K Context

Unsloth provides optimized kernels that can reduce memory usage by up to 50%:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen3-8B",
    max_seq_length=32768,
    load_in_4bit=True,
    dtype=None,  # Auto-detect
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Optimized checkpointing
)
```

**Unsloth Memory Savings:**
| Configuration | Standard | With Unsloth | Savings |
|---------------|----------|--------------|---------|
| 32K context | ~60 GB | ~40-45 GB | ~25-33% |

### 4.6 GPU Compatibility Matrix

| GPU | VRAM | Flash Attn 2 | BF16 | 2K Context | 32K Context |
|-----|------|--------------|------|------------|-------------|
| T4 | 16 GB | ❌ No | ❌ No | ✅ Yes | ❌ No |
| V100 | 16-32 GB | ❌ No | ❌ No | ✅ Yes | ❌ No |
| A10 | 24 GB | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| L4 | 24 GB | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| A100 40GB | 40 GB | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Tight |
| **A100 80GB** | 80 GB | ✅ Yes | ✅ Yes | ✅ Yes | ✅ **Yes** |
| H100 | 80 GB | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

**Key Insight:** For 32K context training, **A100 80GB is the minimum viable option**. T4/V100 won't work regardless of optimizations.

---

## 5. Azure Infrastructure & Cost Estimates

### 5.1 Recommended Azure GPU Instances

#### **For Training Qwen3-8B**

| Instance | GPUs | GPU Memory | On-Demand $/hr | Spot $/hr | 2K Context | 32K Context |
|----------|------|------------|----------------|-----------|------------|-------------|
| **NC8as T4 v3** | 1x T4 | 16 GB | $0.75 | $0.23 | ✅ Yes | ❌ No |
| **NC16as T4 v3** | 1x T4 | 16 GB | $1.20 | $0.36 | ✅ Yes | ❌ No |
| **NC24ads A100 v4** | 1x A100 | 80 GB | $3.67 | $1.15 | ✅ Overkill | ✅ **Required** |
| **ND96amsr A100 v4** | 8x A100 | 640 GB | $32.77 | $8.52 | ✅ Overkill | ✅ Multi-GPU |
| **NC40ads H100 v5** | 1x H100 | 80 GB | $6.98 | ~$2.10 | ✅ Overkill | ✅ Fastest |

#### **Instance Selection Guide for Qwen3-8B**

| Your Use Case | Context Length | Recommended Instance | Spot $/hr |
|---------------|----------------|----------------------|-----------|
| Budget training | 2K-4K | NC8as T4 v3 | $0.23 |
| Balanced training | 2K-8K | NC16as T4 v3 | $0.36 |
| Long context training | 16K-32K | NC24ads A100 v4 | $1.15 |
| Maximum speed (32K) | 32K | NC40ads H100 v5 | $2.10 |

#### **For Inference**

| Instance | GPUs | GPU Memory | On-Demand $/hr | Spot $/hr | Use Case |
|----------|------|------------|----------------|-----------|----------|
| **NC8as T4 v3** | 1x T4 | 16 GB | $0.75 | $0.23 | INT4 quantized inference |
| **NC24ads A100 v4** | 1x A100 | 80 GB | $3.67 | $1.15 | Production inference, long context |
| **NC40ads H100 v5** | 1x H100 | 80 GB | $6.98 | ~$2.10 | High-throughput inference |

---

### 5.2 Storage Costs

#### **Azure Blob Storage Pricing**

For a single model like Qwen3-8B, storage costs are **negligible** (~$4-10/month). Use **Hot tier only** during active development - you can migrate to Cool/Archive later if you accumulate many experiments.

| Tier | Price/GB/month | When to Use |
|------|----------------|-------------|
| **Hot** | $0.018 | Active development (recommended) |
| **Cool** | $0.010 | Archived datasets you access monthly |
| **Archive** | $0.001 | Long-term backup (hours to retrieve) |

#### **Storage Requirements for Qwen3-8B**

| Item | Size | Monthly Cost (Hot) |
|------|------|-------------------|
| Model weights (BF16) | 16 GB | $0.29 |
| Model weights (INT4 quantized) | 4-5 GB | $0.09 |
| QLoRA adapter | 100-500 MB | $0.01 |
| Training dataset (typical SFT) | 10-50 GB | $0.90 |
| Checkpoints (keep 3) | 150 GB | $2.70 |
| **Total** | **~220 GB** | **~$4/month** |

**Key Insight:** Storage is not a significant cost driver. Focus your budget on GPU compute.

---

### 5.3 Monthly Cost Estimates for Qwen3-8B

#### **Scenario A: Short Context (2K-4K) - Budget Development**

| Resource | Configuration | Hours/Month | Cost |
|----------|---------------|-------------|------|
| **Training (QLoRA)** | NC8as T4 v3 (Spot) | 80 hrs | $18 |
| **Inference Testing** | NC8as T4 v3 (Spot) | 60 hrs | $14 |
| **Developer VMs** | 2x D4s v3 | 400 hrs | $70 |
| **Storage** | Hot tier, 220GB | - | $4 |
| **Networking** | Egress ~50 GB | - | $5 |
| **Misc** | Monitor, Key Vault | - | $10 |
| **Total** | | | **~$120/month** |
| **With 25% Buffer** | | | **$150/month** |

#### **Scenario B: Long Context (32K) - Development**

| Resource | Configuration | Hours/Month | Cost |
|----------|---------------|-------------|------|
| **Training (QLoRA 32K)** | NC24ads A100 v4 (Spot) | 120 hrs | $138 |
| **Inference Testing** | NC24ads A100 v4 (Spot) | 60 hrs | $69 |
| **Developer VMs** | 2x D4s v3 | 400 hrs | $70 |
| **Storage** | Hot tier, 250GB | - | $5 |
| **Networking** | Egress ~50 GB | - | $5 |
| **Misc** | Monitor, Key Vault | - | $10 |
| **Total** | | | **~$297/month** |
| **With 20% Buffer** | | | **$360/month** |

#### **Scenario C: Production Inference (Qwen3-8B)**

| Configuration | Instance | Hours/Month | Monthly Cost |
|---------------|----------|-------------|--------------|
| **Budget (INT4 quantized)** | NC8as T4 v3 (Spot) | 720 | $166 |
| **Budget (INT4 quantized)** | NC8as T4 v3 (On-demand) | 720 | $540 |
| **Standard (BF16)** | NC24ads A100 v4 (Spot) | 720 | $828 |
| **Standard (BF16)** | NC24ads A100 v4 (Reserved 1-yr) | 720 | ~$1,850 |
| **High Performance** | NC40ads H100 v5 (Spot) | 720 | ~$1,512 |

**Production Total (with infrastructure):**

| Tier | Inference | Storage | Network | LB | Misc | **Total** |
|------|-----------|---------|---------|-----|------|-----------|
| Budget (T4 Spot) | $166 | $4 | $44 | $20 | $15 | **$250/month** |
| Standard (A100 Spot) | $828 | $4 | $44 | $20 | $15 | **$910/month** |
| Enterprise (A100 Reserved) | $1,850 | $4 | $87 | $20 | $30 | **$1,990/month** |

---

### 5.4 Cost Comparison Summary

| Scenario | Context | GPU | Monthly Cost |
|----------|---------|-----|--------------|
| **Development (Budget)** | 2K-4K | T4 16GB | **$150** |
| **Development (32K Context)** | 32K | A100 80GB | **$360** |
| **Production (Budget)** | Any (INT4) | T4 16GB | **$250** |
| **Production (Standard)** | Any | A100 80GB | **$910** |
| **Production (Enterprise)** | Any | A100 Reserved | **$2,000** |

---

### 5.5 Cost Optimization Strategies

| Strategy | Savings | Trade-off |
|----------|---------|-----------|
| **Spot Instances** | 60-70% | Possible interruptions |
| **Use T4 for 2K context** | 80% vs A100 | Limited to short context |
| **INT4 Quantization** | Use T4 instead of A100 | Minor quality loss |
| **1-Year Reserved** | 30-40% | Commitment required |
| **Scheduled scaling** | 30-50% | Off-peak only |

---

### 5.6 Detailed Monthly Budget Request for Qwen3-8B

#### **Option A: Short Context (2K-4K) - 2 Developers**

```
AZURE MONTHLY BUDGET REQUEST - QWEN3-8B (2K CONTEXT)
====================================================

Compute - Training (QLoRA):
  NC8as T4 v3 (1x T4 16GB, Spot)
  - 80 hours/month @ $0.23/hr = $18

Compute - Inference Testing:
  NC8as T4 v3 (Spot)
  - 60 hours/month @ $0.23/hr = $14

Compute - Developer VMs:
  2x Standard D4s v3 (4 vCPU, 16 GB RAM)
  - 2 VMs × 200 hrs/month × $0.175/hr = $70

Storage:
  Azure Blob Storage (Hot): 220 GB @ $0.018 = $4

Networking:
  Egress: 50 GB @ $0.087 = $5

Miscellaneous:
  Azure Monitor, Key Vault = $10

====================================================
TOTAL MONTHLY ESTIMATE: $121
BUFFER (25%): $29
====================================================
REQUESTED MONTHLY BUDGET: $150
====================================================

QUARTERLY BUDGET (3 months): $450
```

#### **Option B: Long Context (32K) - 2 Developers**

```
AZURE MONTHLY BUDGET REQUEST - QWEN3-8B (32K CONTEXT)
=====================================================

Compute - Training (QLoRA 32K):
  NC24ads A100 v4 (1x A100 80GB, Spot)
  - 120 hours/month @ $1.15/hr = $138
  NOTE: A100 80GB required for 32K context

Compute - Inference Testing:
  NC24ads A100 v4 (Spot)
  - 60 hours/month @ $1.15/hr = $69

Compute - Developer VMs:
  2x Standard D4s v3 (4 vCPU, 16 GB RAM)
  - 2 VMs × 200 hrs/month × $0.175/hr = $70

Storage:
  Azure Blob Storage (Hot): 250 GB @ $0.018 = $5

Networking:
  Egress: 50 GB @ $0.087 = $5

Miscellaneous:
  Azure Monitor, Key Vault = $10

=====================================================
TOTAL MONTHLY ESTIMATE: $297
BUFFER (20%): $63
=====================================================
REQUESTED MONTHLY BUDGET: $360
=====================================================

QUARTERLY BUDGET (3 months): $1,080
```

#### **Production Phase (per month)**

```
AZURE MONTHLY BUDGET REQUEST - PRODUCTION
=========================================

Option A: Budget Production (INT4 Quantized on T4)
--------------------------------------------------
Compute - Inference (24/7):
  NC8as T4 v3 (Spot with fallback)
  - Primary: 720 hrs × $0.23/hr = $166

Option B: Standard Production (A100)
------------------------------------
Compute - Inference (24/7):
  NC24ads A100 v4 (Spot)
  - 720 hrs × $1.15/hr = $828

Common Infrastructure:
  Developer VMs: 2x D4s v3 = $70
  Storage (Hot): 250 GB = $5
  Networking (1TB egress) = $87
  Load Balancer = $20
  Monitoring = $15

=========================================
TOTAL (Budget T4): $363/month → Request $400
TOTAL (Standard A100): $1,025/month → Request $1,200
=========================================
```

---

## 6. Recommended Architecture

### 6.1 High-Level Architecture for Qwen3-8B

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AZURE CLOUD                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────┐    ┌───────────────────┐    ┌────────────┐  │
│  │   Training        │    │   Inference       │    │  Storage   │  │
│  │   Pipeline        │    │   Service         │    │            │  │
│  └─────────┬─────────┘    └─────────┬─────────┘    └─────┬──────┘  │
│            │                        │                     │         │
│  ┌─────────▼─────────┐    ┌─────────▼─────────┐    ┌─────▼──────┐  │
│  │  2K Context:      │    │  Production:      │    │ Blob Store │  │
│  │  NC8as T4 v3      │    │  NC8as T4 (INT4)  │    │ (Hot tier) │  │
│  │  (1x T4 16GB)     │    │  or               │    │            │  │
│  │                   │    │  NC24ads A100     │    │ - Qwen3-8B │  │
│  │  32K Context:     │    │  (BF16/32K)       │    │   (~20GB)  │  │
│  │  NC24ads A100 v4  │    │                   │    │ - Dataset  │  │
│  │  (1x A100 80GB)   │    │  vLLM + EAGLE-3   │    │ - Checkpts │  │
│  │                   │    │  + PagedAttention │    │            │  │
│  │  Axolotl/Unsloth  │    │  + Quantization   │    │  ~$4/month │  │
│  └───────────────────┘    └───────────────────┘    └────────────┘  │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    Orchestration Layer                       │   │
│  │  Azure Kubernetes Service (AKS) / Azure ML                   │   │
│  │  - Auto-scaling    - Model Registry    - Monitoring          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Technology Stack Summary for Qwen3-8B

| Layer | Technology | Status |
|-------|------------|--------|
| **Model** | Qwen3-8B | Ready to use |
| **Training Framework** | Axolotl / Unsloth + HuggingFace PEFT | Ready to use |
| **Training Method** | QLoRA (4-bit NF4) | Ready to use |
| **Training GPU (2K)** | T4 16GB | Ready to use |
| **Training GPU (32K)** | A100 80GB (required) | Ready to use |
| **Inference Framework** | vLLM | Ready to use |
| **Inference Optimization** | PagedAttention + Continuous Batching | Ready to use |
| **Quantization** | INT4/FP8 for inference | Ready to use |
| **Speculative Decoding** | EAGLE-3 with Qwen3-0.6B draft | Partially ready |
| **Orchestration** | Azure Kubernetes Service | Ready to use |
| **Monitoring** | Azure Monitor + Prometheus | Ready to use |

### 6.3 Implementation Priority for Qwen3-8B

| Priority | Task | GPU Required | Custom Dev |
|----------|------|--------------|------------|
| P0 | Deploy vLLM with Qwen3-8B | T4 (INT4) or A100 (BF16) | ❌ No |
| P0 | Enable continuous batching + PagedAttention | Any | ❌ No |
| P1 | Set up QLoRA training pipeline | T4 (2K) / A100 (32K) | ❌ No |
| P1 | Implement INT4 quantization for T4 deployment | T4 | ❌ No |
| P2 | Enable speculative decoding (Qwen3-0.6B draft) | A100 | ❌ No |
| P2 | Train EAGLE-3 draft head | A100 | ⚠️ Training only |
| P3 | Evaluate TensorRT-LLM for max throughput | A100/H100 | ❌ No |

### 6.4 Quick Start Commands

**Training (2K Context on T4):**
```bash
# Using Unsloth for optimized training
pip install unsloth
python train.py --model Qwen/Qwen3-8B --max_seq_length 2048 --load_in_4bit
```

**Training (32K Context on A100):**
```bash
# Using Axolotl with Flash Attention
pip install axolotl[flash-attn]
accelerate launch -m axolotl.cli.train config_32k.yaml
```

**Inference (Production):**
```bash
# vLLM with INT4 quantization (works on T4)
pip install vllm
vllm serve Qwen/Qwen3-8B-AWQ --quantization awq --max-model-len 4096

# vLLM with BF16 (requires A100)
vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 32768
```

---

## Sources

### Small Language Models
- [BentoML - Best Open-Source Small Language Models](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
- [DataCamp - Top 15 Small Language Models 2026](https://www.datacamp.com/blog/top-small-language-models)
- [Qwen3 Official Blog](https://qwenlm.github.io/blog/qwen3/)
- [Qwen3 GitHub](https://github.com/QwenLM/Qwen3)
- [Data Science Dojo - Qwen Models Guide](https://datasciencedojo.com/blog/the-evolution-of-qwen-models/)

### Inference Optimization
- [NVIDIA - Speculative Decoding Introduction](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)
- [BentoML - 3x Faster Inference with Speculative Decoding](https://www.bentoml.com/blog/3x-faster-llm-inference-with-speculative-decoding)
- [EAGLE-3 Paper](https://arxiv.org/abs/2503.01840)
- [EAGLE GitHub](https://github.com/SafeAILab/EAGLE)
- [Red Hat - EAGLE-3 with vLLM](https://developers.redhat.com/articles/2025/07/01/fly-eagle3-fly-faster-inference-vllm-speculative-decoding)
- [vLLM vs TensorRT-LLM Comparison](https://medium.com/@hadiyolworld007/vllm-vs-tensorrt-llm-the-2025-inference-smackdown-55adca681bf8)
- [MarkTechPost - Top 6 Inference Runtimes](https://www.marktechpost.com/2025/11/07/comparing-the-top-6-inference-runtimes-for-llm-serving-in-2025/)
- [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)

### Training Optimization
- [HuggingFace PEFT](https://github.com/huggingface/peft)
- [TensorBlue - LLM Fine-tuning Guide](https://tensorblue.com/blog/llm-fine-tuning-complete-guide-tutorial-2025)
- [Analytics Vidhya - LoRA and QLoRA](https://www.analyticsvidhya.com/blog/2023/08/lora-and-qlora/)
- [Cameron Wolfe - PEFT Methods](https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft)

### Azure Pricing
- [Azure Blob Storage Pricing](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/)
- [Vantage - NC24ads A100 v4 Pricing](https://instances.vantage.sh/azure/vm/nc24ads-v4)
- [Vantage - ND96isr H100 v5 Pricing](https://instances.vantage.sh/azure/vm/nd96isrh100-v5)
- [Cyfuture - Azure ND H100 v5 Pricing](https://cyfuture.cloud/kb/gpu/azure-nd-h100-v5-pricing-per-hour-complete-cost-breakdown)
- [H100 Cloud Pricing Comparison](https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison)

---

*Document generated: January 2026*
