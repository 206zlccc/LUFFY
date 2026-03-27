# LUFFY Development Repository

> 🚧 **Development Branch** - This is the main development repository for LUFFY (Learning to Reason Under Off‑Policy Guidance)

## About LUFFY

LUFFY is a reinforcement learning framework that bridges the gap between zero-RL and imitation learning by incorporating off-policy reasoning traces into the training process. This repository contains the core implementation and development work.

## 🔧 Development Status

This repository is under active development. Many features are currently being implemented or need refactoring.

## 🚀 Quick Start

⚠️ **Note**: This development version has incomplete implementations. Many features are marked as TODO and need to be completed before production use.

```bash
# Clone the repository
git clone <repository-url>
cd LUFFY

# Install dependencies
pip install -r luffy/requirements.txt

# Note: Some functionality is incomplete - check TODO list below for details
```

## 📁 Repository Structure

```
LUFFY/
├── luffy/                 # Core framework
│   ├── deepscaler/        # Scaling utilities (⚠️ API integration needed)
│   ├── verl/              # RL training components (⚠️ Some features incomplete)
│   └── ...
├── data/                  # Training data and scripts
├── eval_scripts/          # Evaluation utilities
├── exp_scripts/           # Experiment scripts
└── README.md              # This file
```

## ⚠️ Development Notes

- This is a **development version** with incomplete implementations
- Many functions contain TODO markers indicating pending work
- API integrations (OpenAI, Gemini) are currently placeholder implementations
- FSDP and distributed training features need completion


### 🔴 High Priority TODOs

- **API Integration**: OpenAI and Gemini API implementations need completion
- **Reward System**: Parallel processing and validation for reward computation  
- **FSDP Training**: Model loading and distributed training setup
- **Data Processing**: Batch dimension operations and tensor reshaping

### 📝 Complete TODO List

- [ ] **./luffy/verl/tests/model/test_transformer.py:22** - add more models for test
- [ ] **./luffy/verl/tests/model/test_transformers_ulysses.py:34** - add more models for test
- [ ] **./luffy/verl/verl/mix_src/mix_fsdp_worker.py:54** - support FSDP hybrid shard for larger model
- [ ] **./luffy/verl/verl/mix_src/mix_fsdp_worker.py:123** - 1. support create from random initialized model. 2. Support init with FSDP directly
- [ ] **./luffy/verl/verl/mix_src/mix_fsdp_worker.py:252** - support FSDP hybrid shard for larger model
- [ ] **./luffy/verl/verl/models/registry.py:21** - HF may supported more than listed here, we should add more after testing
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py:64** - delete the unused args
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/arg_utils.py:147** - Support fine-grained seeds (e.g., seed per request).
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py:237** - maybe we can hack the autoregressive logics without only apply post process for better performance
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py:241** - we can optimize it by making the dataloader yield List[int] without padding.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm.py:257** - can be optimzied by rewrite the Sampler._get_logprobs() logits
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py:99** - Print more configs in debug mode.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/llm_engine_sp.py:112** - maybe we can choose init here or from arguments
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/model_loader.py:224** - Change the get_logits part to a separate stage.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/tokenizer.py:56** - the lora tokenizer is also passed, but may be different
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py:209** - Profile swapping overhead and optimize if needed.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_3_1/worker.py:291** - maybe we should also flag the megatron is initialized
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py:109** - delete the unused args
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/arg_utils.py:192** - Support fine-grained seeds (e.g., seed per request).
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py:268** - maybe we can hack the autoregressive logics without only apply post process for better performance
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py:272** - we can optimize it by making the dataloader yield List[int] without padding.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm.py:288** - can be optimzied by rewrite the Sampler._get_logprobs() logits
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:128** - Print more configs in debug mode.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:143** - maybe we can choose init here or from arguments
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/llm_engine_sp.py:228** - add for verl but we may not tokenizer in Rollout
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py:141** - This is a hack, we need to register the load_weight() func for each model in vllm
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/model_loader.py:226** - This is a hack, we need to register the load_weight() func for each model in vllm
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/model_runner.py:274** - perform sampling on rank 0
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py:62** - verl not support speculative decode now
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/spmd_gpu_executor.py:208** - not implemented async executor yet
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/tokenizer.py:61** - the lora tokenizer is also passed, but may be different
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/worker.py:30** - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_4_2/worker.py:270** - check whether need this
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py:143** - delete the unused args
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/arg_utils.py:226** - Support fine-grained seeds (e.g., seed per request).
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm.py:221** - can be optimzied by rewrite the Sampler._get_logprobs() logits
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py:143** - Print more configs in debug mode.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py:160** - maybe we can choose init here or from arguments
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/llm_engine_sp.py:262** - add for verl but we may not tokenizer in Rollout
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py:152** - This is a hack, we need to register the load_weight() func for each model in vllm
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/model_loader.py:239** - This is a hack, we need to register the load_weight() func for each model in vllm
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/parallel_state.py:94** - deviate from the v0.5.4, not pp now
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py:65** - verl not support speculative decode now
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/spmd_gpu_executor.py:243** - not implemented async executor yet
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/tokenizer.py:61** - the lora tokenizer is also passed, but may be different
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py:29** - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py:103** - set correct model runner class
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_5_4/worker.py:301** - check whether need this
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm.py:186** - can be optimzied by rewrite the Sampler._get_logprobs() logits
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py:174** - Print more configs in debug mode.
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/llm_engine_sp.py:336** - add for verl but we may not tokenizer in Rollout
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py:170** - This is a hack, we need to register the load_weight() func for each model in vllm
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/model_loader.py:273** - This is a hack, we need to register the load_weight() func for each model in vllm
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/parallel_state.py:97** - deviate from the v0.5.4, not pp now
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py:73** - verl not support speculative decode now
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/spmd_gpu_executor.py:246** - not implemented async executor yet
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py:33** - check why vllm has similar file in vllm.model_executor.parallel_utils.parallel_state
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py:110** - set correct model runner class
- [ ] **./luffy/verl/verl/third_party/vllm/vllm_v_0_6_3/worker.py:311** - check whether need this
- [ ] **./luffy/verl/verl/utils/megatron_utils.py:202** - check how to disable megatron timers
- [ ] **./luffy/verl/verl/workers/actor/megatron_actor.py:158** - actually, this function should only return log_prob and this logic should be handled by user outside
- [ ] **./luffy/verl/verl/workers/fsdp_workers.py:88** - support FSDP hybrid shard for larger model
- [ ] **./luffy/verl/verl/workers/fsdp_workers.py:157** - 1. support create from random initialized model. 2. Support init with FSDP directly
- [ ] **./luffy/verl/verl/workers/fsdp_workers.py:278** - support FSDP hybrid shard for larger model
- [ ] **./luffy/verl/verl/workers/fsdp_workers.py:811** - we may need to extract it to dp_reward_model.py
- [ ] **./luffy/verl/verl/workers/megatron_workers.py:106** - Currently, we only support reference model param offload
- [ ] **./luffy/verl/verl/workers/megatron_workers.py:444** - support critic model offload
- [ ] **./luffy/verl/verl/workers/sharding_manager/megatron_vllm.py:253** - this may not be true for FSDP -> vLLM
- [ ] **./luffy/verl/verl/workers/sharding_manager/megatron_vllm.py:273** - currently, the implementation is adhoc. We can move this function to the model
## 🤝 Contributing

1. Pick a TODO item from the list above
2. Implement the functionality
3. Test your implementation
4. Update this README when TODOs are completed

