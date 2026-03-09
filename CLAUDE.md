# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **Never run tests on this server.** When tests need to be run, ask the user to run them on a remote server instead.
- **Never add Anthropic-related information in git commit messages.** No `Co-Authored-By: Claude` or similar attribution lines.

## Project Overview

slime is an LLM post-training framework for RL scaling that connects Megatron (training) with SGLang (inference/rollout). It powers GLM-4.5/4.6/4.7 and supports Qwen3, DeepSeek V3/R1, and Llama 3 models.

**Key Papers & Blogs**:
- Vision: [slime: An SGLang-Native Post-Training Framework for RL Scaling](https://lmsys.org/blog/2025-07-09-slime/)
- Agentic training: [Agent-Oriented Design: An Asynchronous and Decoupled Framework for Agentic RL](https://www.notion.so/Agent-Oriented-Design-An-Asynchronous-and-Decoupled-Framework-for-Agentic-RL-2278e692d081802cbdd5d37cef76a547)
- Release notes: [v0.1.0: Redefining High-Performance RL Training Frameworks](https://thudm.github.io/slime/blogs/release_v0.1.0.html)

## Architecture

The framework has three core modules:
- **training (Megatron/FSDP)**: Main training process, reads data from Data Buffer, syncs parameters to rollout after training
- **rollout (SGLang + router)**: Generates new data including rewards/verifier outputs, stores in Data Buffer
- **data buffer**: Bridge module managing prompt initialization, custom data, and rollout generation

Training loop: `Data Sampling (Rollout) -> Weight Update (Training) -> Weight Sync to SGLang`

**Data Flow**:
1. Data Buffer loads prompts from disk or custom sources
2. Rollout engines generate completions via SGLang
3. Rewards computed (via reward models or verifiers)
4. Training actor consumes batches, updates weights
5. Updated weights synced back to rollout engines

## Common Commands

### Installation
```bash
pip install -e . --no-deps
```

### Code Style (pre-commit)
```bash
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always
```

### Running Tests
```bash
pytest                           # Run all tests
pytest tests/test_file.py        # Run specific test file
pytest -m unit                   # Run only unit tests
pytest -m "not system"           # Exclude system tests
pytest -v -s                     # Verbose with stdout
```

### Model Weight Conversion
```bash
# HuggingFace to Megatron torch_dist format
# 1. Load model config first
cd /root/slime
source scripts/models/glm4-9B.sh  # or qwen3-4B.sh, etc.

# 2. Convert weights
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} --hf-checkpoint <hf_path> --save <output_path>

# Megatron to HuggingFace format
PYTHONPATH=/root/Megatron-LM python tools/convert_torch_dist_to_hf.py \
    --input-dir <megatron_ckpt>/iter_xxx/ --output-dir <hf_output> --origin-hf-dir <original_hf>
```

### Training Workflow
```bash
# 1. Start Ray cluster (required for distributed training)
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# 2. Submit training job via Ray
ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
    -- python3 train.py <args>

# 3. Monitor training (check Ray dashboard or logs)
# Ray dashboard: http://<MASTER_ADDR>:8265

# 4. Stop Ray when done
ray stop --force
```

Example training scripts: `scripts/run-*.sh` (e.g., `scripts/run-qwen3-4B.sh`, `scripts/run-glm4-9B.sh`)

### Quick Debugging
```bash
# Kill all slime-related processes (before restarting)
pkill -9 sglang
pkill -9 ray
pkill -9 python

# Check GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Auto-refresh every 1 second

# Check Ray cluster status
ray status

# Tail logs in real-time
tail -f /tmp/ray/session_latest/logs/*.log
```

## Key Source Directories

### Core Modules
- `slime/backends/megatron_utils/`: Megatron training backend
  - `actor.py`: Main training actor (forward/backward passes, optimizer steps)
  - `checkpoint.py`: Checkpoint loading/saving for Megatron format
  - `model.py`: Model provider for various architectures
  - `update_weight/`: Weight synchronization from training to rollout
- `slime/backends/fsdp_utils/`: FSDP training backend (experimental, alternative to Megatron)
- `slime/backends/sglang_utils/`: SGLang inference engine integration
  - `arguments.py`: SGLang-specific argument handling
  - Engine initialization and management

### Rollout & Rewards
- `slime/rollout/`: Rollout generation orchestration
  - `rm_hub/`: Reward model implementations (DeepScaler, custom RMs)
  - `rollout_function.py`: Main rollout generation logic
  - Dynamic sampling filters (e.g., DAPO-style filtering)
- `slime/router/`: Request routing with middleware support
  - Handles load balancing across multiple SGLang instances

### Distributed Execution
- `slime/ray/`: Ray actors for distributed coordination
  - `actor_rollout.py`: Rollout actor managing SGLang engines
  - `actor_training.py`: Training actor wrapper
  - `placement_group.py`: GPU resource allocation
  - `coordinator.py`: High-level training/rollout coordination

### Configuration & Utilities
- `slime/utils/arguments.py`: All slime-specific CLI arguments (see Arguments System below)
- `slime/utils/eval_config.py`: Evaluation dataset configuration
- `slime/utils/logging_utils.py`: Logging setup
- `slime_plugins/`: Model-specific plugins (GLM4, Qwen3Next, MIMO, DeepSeek)
- `tools/`: Weight conversion utilities (HF ↔ Megatron)

### Examples & Tests
- `examples/`: Real-world use cases (multi-turn, VLM, agentic RL, async training)
- `tests/`: Unit and integration tests
- `scripts/`: Training scripts for various models (GLM4, Qwen3, DeepSeek, etc.)

## Arguments System

Three categories of arguments:
1. **Megatron args**: Standard Megatron flags (e.g., `--tensor-model-parallel-size 2`)
2. **SGLang args**: Prefixed with `--sglang-` (e.g., `--sglang-mem-fraction-static 0.8`)
3. **slime args**: Defined in `slime/utils/arguments.py`

### Critical slime Arguments

**Resource Allocation**:
- `--train-backend`: `megatron` (default) or `fsdp`
- `--colocate`: Share GPUs between training and inference (enables offloading)
- `--actor-num-nodes`, `--actor-num-gpus-per-node`: Training cluster size
- `--rollout-num-gpus`: Total GPUs for inference (ignored if colocated)
- `--rollout-num-gpus-per-engine`: Tensor parallelism for each SGLang engine
- `--offload`, `--offload-train`, `--offload-rollout`: CPU offloading control

**Data Flow Control**:
- `--rollout-batch-size`: Batch size for rollout generation
- `--n-samples-per-prompt`: Multiple completions per prompt (for best-of-N, etc.)
- `--global-batch-size`: Training batch size
- `--num-rollout`: Total number of rollout samples per iteration
- `--prompt-data`: Path to prompt dataset (JSONL format)
- `--input-key`, `--label-key`: Keys in JSONL for prompts and labels
- `--apply-chat-template`: Apply chat template to prompts

**RL Algorithms**:
- `--advantage-estimator`: `grpo`, `gspo`, `reinforce_plus_plus`, `ppo`, `dpo`
- `--kl-coef`: KL divergence coefficient for policy regularization
- `--entropy-coef`: Entropy bonus coefficient
- `--gamma`: Discount factor for GAE/rewards

**Reward & Rollout Customization**:
- `--rm-type`: Built-in reward model type (e.g., `deepscaler`, `prm`)
- `--custom-rm-path`: Path to custom reward function (format: `module.file:function_name`)
- `--rollout-function-path`: Custom rollout generation logic
- `--custom-generate-function-path`: Custom per-sample generation (for multi-turn)
- `--dynamic-sampling-filter-path`: Dynamic sampling filter (e.g., DAPO-style)
- `--custom-loss-function-path`: Custom loss computation

**Checkpointing**:
- `--hf-checkpoint`: HuggingFace checkpoint path (for conversion)
- `--load`: Load Megatron checkpoint directory
- `--save`: Save Megatron checkpoint directory
- `--ref-load`: Reference model checkpoint for KL divergence
- `--save-interval`: Save checkpoint every N iterations

**Evaluation**:
- `--eval-on-the-fly`: Enable evaluation during training
- `--eval-datasets-and-args`: YAML config for evaluation datasets
- `--eval-interval`: Evaluate every N iterations

## Customization Points

Custom functions are specified via path strings (e.g., `module.file:function_name`):

### Custom Reward Models (`--custom-rm-path`)
Define reward computation logic. Signature:
```python
def custom_rm(responses: List[str], prompts: List[str], labels: List[str], **kwargs) -> List[float]:
    """
    Args:
        responses: Generated completions
        prompts: Original prompts
        labels: Ground truth labels (if available)
    Returns:
        List of scalar rewards (one per response)
    """
```
Example: `slime.rollout.rm_hub.deepscaler_rm:deepscaler_rm`

### Custom Rollout Functions (`--rollout-function-path`)
Control entire rollout generation process. Signature:
```python
def custom_rollout(
    args, prompts, ref_labels, sglang_endpoints, tokenizer, iteration, **kwargs
) -> Tuple[List[str], List[float], Dict]:
    """
    Args:
        args: Parsed arguments
        prompts: List of prompt strings
        ref_labels: Reference labels (if available)
        sglang_endpoints: List of SGLang engine URLs
        tokenizer: HuggingFace tokenizer
        iteration: Current training iteration
    Returns:
        (responses, rewards, metadata_dict)
    """
```

### Custom Generation Functions (`--custom-generate-function-path`)
Per-sample generation logic (for multi-turn, tree search, etc.). Signature:
```python
def custom_generate(prompt: str, endpoint: str, **kwargs) -> Tuple[str, Dict]:
    """
    Args:
        prompt: Single prompt string
        endpoint: SGLang endpoint URL
    Returns:
        (generated_text, metadata_dict)
    """
```
Example: `examples/multi_agent/multi_agent_rollout:generate_for_single_prompt`

### Dynamic Sampling Filters (`--dynamic-sampling-filter-path`)
Filter or reweight rollout samples before training (DAPO-style). Signature:
```python
def dynamic_filter(
    prompts, responses, rewards, iteration, **kwargs
) -> Tuple[List, List, List]:
    """
    Args:
        prompts, responses, rewards: Rollout data
        iteration: Current iteration
    Returns:
        Filtered (prompts, responses, rewards)
    """
```

### Custom Loss Functions (`--custom-loss-function-path`)
Override default RL loss computation. See `slime/backends/megatron_utils/actor.py` for interface.

## Docker

```bash
docker pull slimerl/slime:latest
docker run --rm --gpus all --ipc=host --shm-size=16g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -it slimerl/slime:latest /bin/bash
```

## Learning Path for New Contributors

### 1. Understand Core Concepts (1-2 hours)
- Read the [vision blog](https://lmsys.org/blog/2025-07-09-slime/) to understand design philosophy
- Review `docs/en/get_started/quick_start.md` for hands-on setup
- Study the architecture diagram in `imgs/arch.png` and README.md

### 2. Code Walkthrough (2-4 hours)
Start with these files in order:

**Entry Point**:
1. `train.py` or `train_async.py`: Main training loop entry
2. `slime/utils/arguments.py`: Understand all configuration options

**Core Training Loop**:
3. `slime/ray/coordinator.py`: High-level orchestration
4. `slime/ray/actor_training.py`: Training actor wrapper
5. `slime/backends/megatron_utils/actor.py`: Actual forward/backward/optimizer logic

**Rollout Pipeline**:
6. `slime/ray/actor_rollout.py`: Rollout actor managing SGLang
7. `slime/rollout/rollout_function.py`: Default rollout generation
8. `slime/rollout/rm_hub/deepscaler_rm.py`: Example reward model

**Weight Sync**:
9. `slime/backends/megatron_utils/update_weight/`: Weight transfer from training to rollout

### 3. Run Examples (2-3 hours)
Work through examples in order of complexity:
1. `tests/test_qwen2.5_0.5B_gsm8k.py`: Simplest single-GPU example
2. `scripts/run-qwen3-4B.sh`: Standard multi-GPU training
3. `examples/multi_agent/`: Multi-turn agentic RL
4. `examples/fully_async/`: Asynchronous training paradigm

### 4. Read Test Cases (1-2 hours)
Tests are excellent documentation:
- `tests/test_*.py`: Integration tests showing full training runs
- `tests/ci/`: CI tests for core functionality
- Look for `pytest.mark` to understand test categories

### 5. Modify & Experiment (ongoing)
Common modifications:
- Create custom reward model: Extend `slime/rollout/rm_hub/`
- Add new model architecture: Add plugin to `slime_plugins/`
- Implement new RL algorithm: Modify `slime/backends/megatron_utils/actor.py`
- Add custom rollout logic: See `examples/*/` for patterns

## Common Development Workflows

### Adding Support for a New Model
1. Create model config in `scripts/models/<model_name>.sh`
2. Add conversion logic in `tools/convert_hf_to_torch_dist.py` (if needed)
3. Create plugin in `slime_plugins/<model_name>/` for model-specific logic
4. Test with small model first (e.g., 0.5B or 4B variant)
5. Add integration test in `tests/test_<model_name>.py`

### Implementing a Custom RL Algorithm
1. Study existing algorithms in `slime/backends/megatron_utils/actor.py`:
   - `compute_grpo_loss()`, `compute_gspo_loss()`, etc.
2. Add new `compute_<your_algo>_loss()` function
3. Register in `--advantage-estimator` argument parsing
4. Test on small dataset first (e.g., GSM8K with 0.5B model)
5. Compare against baseline (GRPO/GSPO)

### Debugging Training Issues
**Tools & Locations**:
- Ray logs: `/tmp/ray/session_latest/logs/`
- Megatron logs: Check stdout or redirect to file
- SGLang logs: Usually in Ray actor logs
- TensorBoard: `tensorboard --logdir tensorboard_log/` (if enabled)

**Common Issues**:
- OOM: Reduce `--rollout-batch-size` or enable `--offload`
- Slow rollout: Increase `--rollout-num-gpus` or tune SGLang args
- NaN loss: Check learning rate, KL coef, or reward scale
- Weight sync failure: Verify checkpoint paths and Ray connectivity

**Debug Flags**:
```bash
export PYTHONBUFFERED=16        # Disable output buffering
export CUDA_LAUNCH_BLOCKING=1   # Synchronous CUDA for better errors
export RAY_LOG_TO_STDERR=1      # Ray logs to stderr
```

### Performance Optimization
**Rollout Speed**:
- Tune `--sglang-mem-fraction-static` (default 0.88, lower for more KV cache)
- Increase `--rollout-num-gpus` or `--rollout-num-gpus-per-engine`
- Use `--colocate` to share GPUs (trades memory for hardware efficiency)
- Profile with `examples/profiling/` scripts (if available)

**Training Speed**:
- Enable gradient checkpointing: `--recompute-granularity full --recompute-method uniform`
- Tune micro-batch size: `--micro-batch-size` (balance GPU utilization vs memory)
- Use mixed precision: `--bf16` or `--fp16`
- Profile with PyTorch profiler or Nsight Systems

**Memory Optimization**:
- Enable offloading: `--offload-train` or `--offload-rollout`
- Reduce sequence length: `--rollout-max-response-len`
- Use ZeRO optimizer: Already enabled in Megatron by default

## Contributing Guidelines

### Before Submitting a PR
1. Run pre-commit hooks: `pre-commit run --all-files`
2. Add tests for new features in `tests/`
3. Update documentation if adding new arguments or features
4. Test on small model (0.5B-4B) before claiming support for large models

### Code Style Conventions
- Follow PEP 8 (enforced by pre-commit)
- Use type hints for function signatures
- Add docstrings for public APIs
- Keep functions focused (single responsibility)
- Prefer explicit over implicit (clear argument names)

### Useful Documentation
- Full docs: https://thudm.github.io/slime/
- Quick start: `docs/en/get_started/quick_start.md`
- Usage guide: `docs/en/get_started/usage.md`
- Debugging: `docs/en/developer_guide/debug.md`
- FAQ: `docs/en/get_started/qa.md`
- Platform support: `docs/en/platform_support/`

### Community & Support
- GitHub Issues: https://github.com/THUDM/slime/issues
- DeepWiki Q&A: https://deepwiki.com/THUDM/slime
- Research projects using slime: See README "Projects Built upon slime" section

## Advanced Topics

### Multi-Turn / Agentic RL
See `examples/multi_agent/` and `examples/tau-bench/` for:
- Tool use with MCP protocol
- Multi-turn dialogue optimization
- Tree search integration

### Asynchronous Training
See `examples/fully_async/` and `train_async.py`:
- Decouple rollout from training completely
- Continuous data generation pipeline
- Higher GPU utilization

### VLM (Vision-Language Models)
See `examples/geo3k_vlm/` and `examples/geo3k_vlm_multi_turn/`:
- Image input handling
- Multi-modal reward models
- VLM-specific chat templates

### Low-Precision Training
See `scripts/low_precision/`:
- INT4 quantization-aware training
- FP8 training support
- Mixed precision strategies
