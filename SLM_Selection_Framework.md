# SLM Selection Framework

## Overview
A systematic framework for selecting Small Language Models (<8B parameters) for any use case.

---

## Phase 1: Requirements Analysis

### 1.1 Task Classification Matrix

| Task Category | Sub-Tasks | Key Model Traits Needed |
|---------------|-----------|-------------------------|
| **Conversational** | Customer support, chatbots, Q&A | Instruction following, low latency, context retention |
| **Code Generation** | Autocomplete, debugging, explanation | Code benchmark scores (HumanEval, MBPP), syntax accuracy |
| **Reasoning** | Math, logic, analysis, planning | MATH/GSM8K scores, chain-of-thought capability |
| **Creative** | Writing, summarization, paraphrasing | Fluency, coherence, style adaptability |
| **Multilingual** | Translation, cross-lingual tasks | Language coverage, FLORES scores |
| **Multimodal** | Vision+text, document understanding | Native vision encoder, OCR capability |
| **Domain-Specific** | Medical, legal, financial | Fine-tuning friendliness, base knowledge |

### 1.2 Constraint Assessment Checklist

| Constraint | Option | Recommendation |
|------------|--------|----------------|
| **Context Length** | Short (<4K) | Most models work |
| | Medium (4K-32K) | Check native support |
| | Long (32K-128K) | Qwen3, Phi-4-mini, Llama 3.1 |
| **Latency** | Real-time (<100ms TTFT) | Smaller models, speculative decoding |
| | Interactive (<500ms) | Standard deployment |
| | Batch processing | Throughput over latency |
| **Hardware** | Consumer GPU (8-16GB) | 3-4B models or quantized 7-8B |
| | Workstation (24GB) | Full 7-8B in FP16 |
| | Cloud (40-80GB) | Any model, long context |
| **Licensing** | Full commercial freedom | Apache 2.0, MIT |
| | With attribution | Llama 3 License |
| | Research only | Some Gemma variants (check version) |
| **Deployment** | Edge/mobile | <4B models, GGUF quantization |
| | On-premise | Framework flexibility important |
| | Cloud | All options available |

---

## Phase 2: Model Evaluation Criteria

### 2.1 Scoring Rubric (Weight by Priority)

| Criterion | Weight | How to Evaluate |
|-----------|--------|-----------------|
| **Task Performance** | 30% | Benchmark scores for target task category |
| **Efficiency** | 20% | Params, memory footprint, tokens/sec |
| **Context Handling** | 15% | Native context length, NIAH scores |
| **Fine-tuning Support** | 15% | PEFT compatibility, community adapters |
| **Framework Compatibility** | 10% | vLLM, TensorRT-LLM, ONNX support |
| **Ecosystem Maturity** | 10% | Documentation, community, tutorials |

### 2.2 Benchmark Reference by Task

| Task | Primary Benchmarks | Secondary Benchmarks |
|------|-------------------|---------------------|
| General | MMLU, MT-Bench | HellaSwag, ARC |
| Code | HumanEval, MBPP | DS-1000, CodeContests |
| Reasoning | GSM8K, MATH | ARC-Challenge, BoolQ |
| Multilingual | FLORES, MGSM | XL-Sum, TyDi QA |
| Instruction | IFEval, AlpacaEval | MT-Bench |

---

## Phase 3: Candidate Models (<8B)

### 3.1 Model Comparison Matrix

| Model | Params | Context | License | Strengths | Weaknesses |
|-------|--------|---------|---------|-----------|------------|
| **Qwen3-8B** | 8B | 128K | Apache 2.0 | 119 languages, unified thinking modes, 50% density improvement | At upper size limit |
| **Qwen3-4B** | 4B | 128K | Apache 2.0 | Good balance, long context, multilingual | Less capable than 8B |
| **Phi-4-mini** | 3.8B | 128K | MIT | Best reasoning/code at size, data quality focus | Narrower general knowledge |
| **Llama 3.1-8B** | 8B | 128K | Llama 3 | Largest ecosystem, most tooling | Restrictive license for some |
| **Mistral 7B** | 7.3B | 32K | Apache 2.0 | Proven production use, efficient | Shorter context than others |
| **Gemma 3-4B** | 4B | 128K | Gemma | Multimodal-ready, Google quality | License nuances |
| **Qwen3-30B-A3B** | 30B/3B active | 128K | Apache 2.0 | MoE architecture, high quality, low inference cost | More complex deployment |

### 3.2 Framework Support Matrix

| Model | vLLM | TensorRT-LLM | llama.cpp | ONNX | HF Transformers |
|-------|------|--------------|-----------|------|-----------------|
| Qwen3-8B/4B | Yes | Yes | Yes | Yes | Yes |
| Phi-4-mini | Yes | Yes | Yes | Yes | Yes |
| Llama 3.1-8B | Yes | Yes | Yes | Yes | Yes |
| Mistral 7B | Yes | Yes | Yes | Yes | Yes |
| Gemma 3-4B | Yes | Yes | Yes | Partial | Yes |
| Qwen3-30B-A3B | Yes | Yes | Partial | No | Yes |

---

## Phase 4: Selection Decision Tree

| Primary Use Case | Constraint | Recommended Model(s) |
|------------------|------------|---------------------|
| **Code Generation / Reasoning / Math** | Tight memory (<16GB) | Phi-4-mini (3.8B) |
| | More memory available | Qwen3-8B or Llama 3.1-8B |
| **Multilingual / Global Deployment** | Need 100+ languages | Qwen3-8B (119 languages) |
| | Major languages only | Llama 3.1-8B or Mistral 7B |
| **General Enterprise Chat** | Apache 2.0 required | Qwen3-8B or Mistral 7B |
| | Maximum ecosystem | Llama 3.1-8B |
| | Budget hardware | Mistral 7B or Qwen3-4B |
| **Multimodal (Vision + Text)** | Any | Gemma 3-4B or Qwen3-VL variants |
| **Edge / Mobile Deployment** | Any | Qwen3-4B, Phi-4-mini, or Gemma 3-4B |
| **Maximum Quality + Low Inference Cost** | Any | Qwen3-30B-A3B (MoE: 3B active params) |

---

## Phase 5: Recommended 4-Model Portfolio

### Your Requirements
- **Use Case**: Enterprise chat/support
- **Architecture**: Dense models only (no MoE)
- **Hardware**: Cloud GPU (40-80GB) - A100, H100
- **License**: Flexible (Llama 3 acceptable)

### Portfolio Strategy: Optimize for Enterprise Chat/Support

| Slot | Model | Rationale | Enterprise Chat Strengths |
|------|-------|-----------|---------------------------|
| **1** | **Qwen3-8B** | Best overall, 119 languages for global support, unified thinking modes | Excellent instruction following, handles complex queries, multilingual customer base |
| **2** | **Llama 3.1-8B** | Largest ecosystem, most enterprise integrations, battle-tested | Proven in production chat systems, extensive fine-tuning resources, best tooling |
| **3** | **Mistral 7B** | High throughput (~150 tok/sec), efficient for high-volume support | Cost-effective scaling, reliable responses, good for FAQ-style interactions |
| **4** | **Phi-4-mini (3.8B)** | Smallest footprint, cost optimization tier | Budget tier for simple queries, A/B testing, fallback option |

### Why This Selection for Enterprise Chat

| Model | MT-Bench (Chat Quality) | Instruction Following | Throughput | Cost/1M tokens |
|-------|-------------------------|----------------------|------------|----------------|
| Qwen3-8B | ~8.5 | Excellent | High | Medium |
| Llama 3.1-8B | ~8.3 | Excellent | High | Medium |
| Mistral 7B | ~7.9 | Very Good | Very High | Low |
| Phi-4-mini | ~7.5 | Good | Highest | Lowest |

### Deployment Strategy

| Tier | Use Case | Recommended Model | Rationale |
|------|----------|-------------------|-----------|
| **Tier 1** | Complex queries, VIP customers | Qwen3-8B or Llama 3.1-8B | Highest quality |
| **Tier 2** | Standard support | Mistral 7B | Best throughput/cost |
| **Tier 3** | Simple FAQs, high volume | Phi-4-mini | Lowest cost |

### Alternative: If Multilingual is Critical

Replace Mistral 7B with **Qwen3-4B** for better multilingual support at similar efficiency.

---

## Phase 6: Enterprise Chat Evaluation Criteria

### Key Metrics for Enterprise Chat/Support

| Metric | How to Measure | Target |
|--------|----------------|--------|
| **Instruction Following** | IFEval benchmark | >70% strict accuracy |
| **Conversation Quality** | MT-Bench multi-turn | >7.5 score |
| **Hallucination Rate** | TruthfulQA, manual review | <5% factual errors |
| **Response Latency** | TTFT on A100 | <100ms for chat |
| **Throughput** | Requests/sec under load | >50 req/sec |
| **Multilingual Quality** | FLORES (if needed) | Within 10% of English |

### Enterprise-Specific Validation

| # | Validation Item | Description |
|---|-----------------|-------------|
| 1 | Customer Query Testing | Test with real customer query samples |
| 2 | Tone Consistency | Evaluate professional, helpful tone |
| 3 | Edge Case Handling | Check angry customers, unclear queries |
| 4 | Refusal Behavior | Verify won't answer out-of-scope questions |
| 5 | Context Retention | Test multi-turn conversation memory |
| 6 | Hallucination Rate | Measure on company-specific facts |
| 7 | A/B Testing | Compare against current solution (if exists) |

### Fine-tuning Strategy for Enterprise Chat

1. **Data Requirements**
   - 5K-50K high-quality conversation examples
   - Include: greeting, FAQ, escalation, closing patterns
   - Domain-specific terminology and policies

2. **Recommended Approach**
   - Start with QLoRA (r=16, all layers)
   - Use DPO for tone/style alignment
   - Evaluate after each training run with held-out test set

---

## Quick Reference: Selection by Constraint

| If your top priority is... | Choose |
|---------------------------|--------|
| Best overall quality (<8B) | Qwen3-8B |
| Smallest efficient model | Phi-4-mini (3.8B) |
| Maximum ecosystem/tooling | Llama 3.1-8B |
| Consumer hardware deployment | Mistral 7B |
| Multilingual (100+ langs) | Qwen3-8B |
| Code generation | Phi-4-mini or Qwen3-8B |
| MoE architecture testing | Qwen3-30B-A3B |
| Multimodal capability | Gemma 3-4B |
| Fully open license | Qwen3 (Apache 2.0) or Phi-4 (MIT) |

---

## Implementation Notes for Your Engine

1. **Training Pipeline Considerations**
   - All 4 recommended models support QLoRA/LoRA
   - Unified training config possible via Axolotl/Unsloth
   - Same PEFT approach works across all

2. **Inference Pipeline Considerations**
   - All support vLLM with PagedAttention
   - Quantization (INT4/INT8) available for all
   - Speculative decoding: Use same-family smaller models as drafts

3. **Testing Strategy**
   - Create model-agnostic evaluation harness
   - Benchmark each model on same tasks
   - Compare quality vs latency vs cost tradeoffs

---

