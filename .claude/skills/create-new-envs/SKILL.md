---
name: create-new-envs
description: Create a new gym-like interactive environment integration for slime's multi-turn RL training pipeline. Use when adding environments where an agent takes actions, receives observations, and earns rewards over multiple turns (e.g., web browsing, game playing, robotics simulation, code execution sandboxes).
---

# Create New Environment Integration

Integrate a new interactive environment into slime following the patterns in `examples/android_world/`.

## File Structure

Create the following files under `examples/<your_env>/`:

```
examples/your_env/
  __init__.py
  config.yaml                # Env settings (--custom-config-path)
  prompts.py                 # System prompt + per-turn templates
  env_worker.py              # Raw env interaction (plain class, Ray-remoted in pool)
  env_pool.py                # Async singleton pool of workers
  env_your_env.py            # Per-sample BaseInteractionEnv wrapper
  rollout.py                 # generate() — incremental rollout (1 sample/trajectory)
  rollout_history.py         # generate() — history-based rollout (T samples/trajectory)
  run_your_env.sh            # Shell launch script
  data/
    tasks.jsonl              # Task dataset
  tests/
    test_imports.py
    test_env_pool.py
    test_rollout.py
```

## Data Flow

```
generate() coroutine (one per sample, many concurrent on asyncio event loop)
  ├─ pool.acquire()          → get worker from async queue
  ├─ env.reset(task)         → initialize env with a task
  ├─ loop:
  │    ├─ SGLang generate    → model produces action text
  │    ├─ env.step(action)   → env executes action, returns (obs, done, info)
  │    ├─ format_observation → encode obs into VLM chat message
  │    └─ append tokens      → add observation tokens (loss_mask=0)
  ├─ env.get_reward()        → final reward
  ├─ env.close()             → release worker back to pool
  └─ return Sample(s)
```

## Step 1: BaseInteractionEnv Subclass

**File**: `examples/your_env/env_your_env.py`

Import the base class from `examples/geo3k_vlm_multi_turn/base_env.py`:

```python
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv
```

The base class defines:

```python
class BaseInteractionEnv:
    def reset(self):
        raise NotImplementedError
    def step(self, response_text: str):
        raise NotImplementedError
    def close(self):
        pass
    def format_observation(self, observation: dict) -> dict:
        # Default: extracts multi_modal_data and obs_str
```

Your subclass must implement these methods:

| Method | Signature | Returns | Notes |
|--------|-----------|---------|-------|
| `reset` | `async def reset(self) -> tuple[dict, dict]` | `(observation, info)` | Info dict should include `max_steps`. |
| `step` | `async def step(self, response_text: str) -> tuple[Optional[dict], bool, dict]` | `(observation, done, info)` | `done=True` ends the episode. |
| `get_reward` | `def get_reward(self) -> float` | scalar reward | Called after episode ends. |
| `format_observation` | `def format_observation(self, observation: dict, is_initial: bool = True) -> dict` | chat message dict | Returns `{"role": "user", "content": [...]}`. For VLM, include `{"type": "image", "image": <PIL.Image>}`. |
| `close` | `def close(self) -> None` | — | Release worker to pool. Do NOT destroy the worker. |

**CRITICAL**: If your env uses Ray remote actors, use `asyncio.to_thread(ray.get, ...)` instead of bare `ray.get()`. Bare `ray.get()` blocks the entire asyncio event loop and serializes all concurrent samples:

```python
# WRONG — blocks event loop
def step(self, response_text):
    obs, reward, done, info = ray.get(self.worker_ref.step.remote(response_text))

# CORRECT — offloads to thread pool
async def step(self, response_text):
    obs, reward, done, info = await asyncio.to_thread(
        ray.get, self.worker_ref.step.remote(response_text)
    )
```

Example (modeled on `examples/android_world/env_android_world.py`):

```python
import asyncio
from typing import Any, Optional
import ray
from examples.geo3k_vlm_multi_turn.base_env import BaseInteractionEnv

class YourEnv(BaseInteractionEnv):
    def __init__(self, worker_ref, worker_id, pool, task_name, max_turns, **kwargs):
        self.worker_ref = worker_ref
        self.worker_id = worker_id
        self.pool = pool
        self.task_name = task_name
        self.max_turns = max_turns
        self.turn = 0
        self.final_reward = 0.0
        self._closed = False

    async def reset(self) -> tuple[dict, dict]:
        obs, info = await asyncio.to_thread(
            ray.get, self.worker_ref.reset.remote(self.task_name)
        )
        return obs, info

    async def step(self, response_text: str) -> tuple[Optional[dict], bool, dict]:
        self.turn += 1
        obs, step_reward, done, info = await asyncio.to_thread(
            ray.get, self.worker_ref.step.remote(response_text)
        )
        if done:
            self.final_reward = step_reward
        return obs, done, info

    def get_reward(self) -> float:
        return self.final_reward

    def format_observation(self, observation: dict, is_initial: bool = True) -> dict:
        content = []
        if observation.get("image") is not None:
            content.append({"type": "image", "image": observation["image"]})
        if is_initial:
            text = INITIAL_TEMPLATE.format(task=observation.get("task", ""))
        else:
            text = STEP_TEMPLATE.format(step=self.turn, max_steps=self.max_turns)
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    def close(self):
        if not self._closed and self.pool is not None:
            self.pool.release(self.worker_id)
            self._closed = True
```

## Step 2: Worker

**File**: `examples/your_env/env_worker.py`

A plain Python class wrapping a single environment instance. NOT decorated with `@ray.remote` — that happens in the pool. Runs in its own Ray actor process.

```python
class YourEnvWorker:
    def __init__(self, worker_id: int, **env_kwargs):
        """Heavy setup runs here (inside Ray actor)."""
        self.worker_id = worker_id
        self._initialize_env()

    def reset(self, task_name: str, **kwargs) -> tuple[dict, dict]:
        """Reset env with a new task.
        Returns (observation, info).
        observation: dict with keys format_observation() expects.
        info: dict with at least {"task": str, "max_steps": int}.
        """
        ...

    def step(self, raw_action: str) -> tuple[dict | None, float, bool, dict]:
        """Parse action from model text, execute, return (obs, reward, done, info).
        reward: 0.0 for non-terminal, final score on terminal.
        info: at least {"won": bool}.
        """
        ...

    def close(self):
        """Kill processes, free resources."""
        ...
```

The worker parses the model's raw text into an action. See `examples/android_world/env_worker.py:182` (`parse_ui_action_from_response`) for an example extracting `<tool_call>` XML.

## Step 3: Worker Pool

**File**: `examples/your_env/env_pool.py`

Singleton managing a fixed set of Ray actor workers. Uses `asyncio.Queue` for async acquire/release. Workers created once, reused across iterations.

```python
import asyncio
import logging
from typing import Any
import ray
from examples.your_env.env_worker import YourEnvWorker

logger = logging.getLogger(__name__)

class YourEnvPool:
    _instance: "YourEnvPool | None" = None
    _lock = asyncio.Lock()

    @classmethod
    async def get_instance(cls, config: dict[str, Any]) -> "YourEnvPool":
        if cls._instance is not None:
            return cls._instance
        async with cls._lock:
            if cls._instance is not None:
                return cls._instance
            pool = cls()
            await pool._initialize(config)
            cls._instance = pool
            return pool

    async def _initialize(self, config: dict[str, Any]) -> None:
        num_workers = config.get("num_workers", 16)
        resources_per_worker = config.get(
            "resources_per_worker", {"num_cpus": 4, "memory": 8 * 1024**3}
        )
        RemoteWorker = ray.remote(**resources_per_worker)(YourEnvWorker)

        self._workers = []
        for i in range(num_workers):
            worker = RemoteWorker.options(scheduling_strategy="SPREAD").remote(
                worker_id=i,
                # pass env-specific config here
            )
            self._workers.append(worker)

        self._available: asyncio.Queue[int] = asyncio.Queue()
        for i in range(num_workers):
            self._available.put_nowait(i)
        self._num_workers = num_workers

    async def acquire(self) -> tuple[Any, int]:
        worker_id = await self._available.get()
        return self._workers[worker_id], worker_id

    def release(self, worker_id: int) -> None:
        self._available.put_nowait(worker_id)

    async def close(self) -> None:
        close_refs = [w.close.remote() for w in self._workers]
        await asyncio.to_thread(ray.get, close_refs)
        for w in self._workers:
            ray.kill(w)
        self._workers.clear()
        type(self)._instance = None
```

Key points:
- `scheduling_strategy="SPREAD"` distributes workers across Ray nodes.
- `num_workers` >= concurrent in-flight samples (`rollout_batch_size * n_samples_per_prompt`).

## Step 4: generate() Rollout Function

**File**: `examples/your_env/rollout.py`

Entry point via `--custom-generate-function-path examples.your_env.rollout.generate`.

### Mode A: Incremental (1 sample per trajectory)

Token layout for a 3-turn episode:

```
[prompt tokens] [model output 1] [env obs 1] [model output 2] [env obs 2] [model output 3]
                 loss_mask=1       mask=0       mask=1           mask=0       mask=1
```

Required imports:

```python
from examples.geo3k_vlm_multi_turn.rollout import (
    _append_to_sample,
    _encode_observation_for_generation,
    _finalize_sample,
    _merge_multimodal_train_inputs,
    _run_inference_step,
    _should_stop_on_finish,
    _update_budget,
    _update_multimodal_state,
)
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
```

Signature and skeleton (see `examples/android_world/rollout.py` for full reference):

```python
async def generate(args: Any, sample: Sample, sampling_params: dict) -> Sample:
    assert not getattr(args, "partial_rollout", False)

    pool = await YourEnvPool.get_instance(vars(args))
    worker_ref, worker_id = await pool.acquire()
    task_name = (sample.metadata or {}).get("task_name")
    max_turns = getattr(args, "max_turns", 15)

    env = YourEnv(worker_ref=worker_ref, worker_id=worker_id,
                  pool=pool, task_name=task_name, max_turns=max_turns)
    sampling_params = sampling_params.copy()

    try:
        state = GenerateState(args)
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        sample.metadata = sample.metadata or {}

        obs, info = await env.reset()
        first_user_message = env.format_observation(obs)

        # Build initial prompt — apply chat template to [system, first_user_message]
        # See _build_initial_prompt in examples/android_world/rollout.py:50
        prompt_text, prompt_ids, current_image_data, mm_inputs, init_mm_train = \
            _build_initial_prompt(state.tokenizer, state.processor,
                                  YOUR_SYSTEM_PROMPT, first_user_message, ...)

        sample.prompt = prompt_text
        sample.multimodal_inputs = mm_inputs
        sample.tokens = list(prompt_ids)

        mm_train_buf: list[dict | None] = []
        if init_mm_train:
            mm_train_buf.append(init_mm_train)
        response_tokens: list[int] = []
        sample.loss_mask = []
        sample.rollout_log_probs = []
        sample.response_length = 0

        max_ctx = getattr(args, "max_context_len", None)
        budget = (max_ctx - len(sample.tokens)) if max_ctx else \
                 sampling_params.get("max_new_tokens")

        for turn_idx in range(max_turns):
            cur_params = sampling_params.copy()
            if budget is not None:
                cur_params["max_new_tokens"] = budget

            response_text, new_tokens, new_logprobs, finish_type = \
                await _run_inference_step(url, sample.tokens, cur_params,
                                          current_image_data, state.tokenizer)

            _append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1)
            budget = _update_budget(budget, len(new_tokens))

            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            step_obs, done, step_info = await env.step(response_text)
            if done or step_obs is None:
                sample.status = Sample.Status.COMPLETED
                break

            next_msg = env.format_observation(step_obs, is_initial=False)
            obs_ids, obs_img, obs_mm, obs_mm_train = _encode_observation_for_generation(
                state.tokenizer, state.processor, next_msg, sample.metadata,
                getattr(args, "apply_chat_template", True),
                getattr(args, "apply_chat_template_kwargs", None))

            bos_id = state.tokenizer.bos_token_id
            if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
                obs_ids = obs_ids[1:]

            _append_to_sample(sample, response_tokens, obs_ids,
                              [0.0]*len(obs_ids), loss_mask_val=0)
            budget = _update_budget(budget, len(obs_ids))
            current_image_data = _update_multimodal_state(
                sample, current_image_data, obs_img, obs_mm, obs_mm_train, mm_train_buf)

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

        sample.reward = env.get_reward()
        return _finalize_sample(sample, state.tokenizer, response_tokens, mm_train_buf)
    finally:
        try:
            env.close()
        except Exception:
            pass
```

You must also implement `_build_initial_prompt()`. See `examples/android_world/rollout.py:50` — it applies the chat template to `[system_message, first_user_message]`, runs the VLM processor if available, and returns `(prompt_text, prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs)`.

### Mode B: History-Based (T samples per trajectory)

**File**: `examples/your_env/rollout_history.py`

Each turn is an independent sample with a self-contained prompt.

```python
async def generate(args, sample, sampling_params, evaluation=False) -> list[Sample] | Sample:
```

Key differences:
- Returns `list[Sample]` during training, single `Sample` during evaluation.
- Each step builds a fresh prompt via `_build_step_prompt()` — no accumulated context.
- Token budget is per-step: `step_budget = max_context_len - len(prompt_ids_for_this_step)`.
- All T samples share the same trajectory-level reward.

Per-step sample construction (from `examples/android_world/rollout_history.py:164`):

```python
def _make_turn_sample(original_sample, prompt_text, prompt_ids, multimodal_inputs,
                      multimodal_train_inputs, response_tokens, response_logprobs, tokenizer):
    return Sample(
        group_index=original_sample.group_index,
        index=original_sample.index,
        label=original_sample.label,
        metadata=original_sample.metadata,
        prompt=prompt_text,
        tokens=list(prompt_ids) + response_tokens,
        loss_mask=[1] * len(response_tokens),
        rollout_log_probs=response_logprobs,
        response=tokenizer.decode(response_tokens, skip_special_tokens=False),
        response_length=len(response_tokens),
        multimodal_inputs=multimodal_inputs,
        multimodal_train_inputs=_merge_multimodal_train_inputs(
            [multimodal_train_inputs] if multimodal_train_inputs else []),
        status=Sample.Status.COMPLETED,
    )
```

See `examples/android_world/rollout_history.py` for the complete implementation.

## Step 5: Prompt Templates

**File**: `examples/your_env/prompts.py`

Define at minimum:

```python
# 1. System prompt — tool definitions, role instructions
YOUR_SYSTEM_PROMPT = """You are a helpful assistant.
# Tools
<tools>
{"type": "function", "function": {"name": "your_tool", ...}}
</tools>
For each function call, return a json object within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json>}
</tool_call>"""

# 2. Initial observation template (first turn, includes task description)
YOUR_INITIAL_TEMPLATE = """
The user query: {task_description}
Before answering, explain your reasoning in <thinking></thinking> tags.
<image>
"""

# 3. Step template (subsequent turns in incremental mode — minimal)
YOUR_STEP_TEMPLATE = """
Step {current_step} of {max_steps}: <image>
"""

# 4. Full history template (for history-based mode)
YOUR_HISTORY_TEMPLATE = """
The user query: {task_description}
Task progress ({step_count} operations completed out of {max_steps}):
{action_history}
Step {current_step}: <image>
"""
```

Design principles:
- Keep step templates minimal in incremental mode — task and prior actions are already in the token sequence.
- Use `<image>` where the VLM processor expects image tokens.
- Include `<thinking>` / `<conclusion>` XML blocks for chain-of-thought + action summary extraction.

## Step 6: Task Data (JSONL)

**File**: `examples/your_env/data/tasks.jsonl`

```json
{"prompt": "Placeholder text", "label": "", "metadata": {"task_name": "TaskIdentifier", "params_idx": 0}}
```

| Key | Description |
|-----|-------------|
| `prompt` | Placeholder — overridden dynamically by `generate()` after `env.reset()`. |
| `label` | Ground truth label. Empty string if reward comes from environment. |
| `metadata.task_name` | Task identifier passed to `worker.reset()`. |
| `metadata.params_idx` | Parameter variation index. |

CLI args: `--prompt-data examples/your_env/data/tasks.jsonl --input-key prompt --label-key label`

## Step 7: Config YAML

**File**: `examples/your_env/config.yaml`

Loaded via `--custom-config-path`. All keys injected into `args`.

```yaml
max_turns: 30
max_context_len: 16384
history_window_size: 10

num_workers: 64
resources_per_worker:
  num_cpus: 4
  memory: 8589934592  # 8 GB

# Environment-specific settings
your_setting: "value"
```

`num_workers` >= concurrent in-flight samples (`rollout_batch_size * n_samples_per_prompt`).

## Step 8: Launch Script

**File**: `examples/your_env/run_your_env.sh`

Critical CLI args (pattern from `examples/android_world/run_android_world_opt.sh`):

```bash
ROLLOUT_ARGS=(
   --prompt-data examples/your_env/data/tasks.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --custom-generate-function-path examples.your_env.rollout.generate
   --custom-config-path examples/your_env/config.yaml
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 8
   --rollout-max-response-len 4096
   --rollout-temperature 0.7
   --num-steps-per-rollout 1
   --balance-data
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.82
   --sglang-cuda-graph-bs 1 2 4 8 16 24 32
)
```

Entry point is always `train.py`:

```bash
ray job submit --address="http://127.0.0.1:8080" \
   --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/"}}' \
   --no-wait -- python train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --rollout-num-gpus 8 \
   ${ROLLOUT_ARGS[@]} ${GRPO_ARGS[@]} ${SGLANG_ARGS[@]} ...
```

Switch rollout modes by changing `--custom-generate-function-path`:
- Incremental: `examples.your_env.rollout.generate`
- History-based: `examples.your_env.rollout_history.generate`

## Step 9: Reward Design

Rewards computed inside `env_worker.py`'s `step()`. No `--rm-type` or `--custom-rm-path` needed.

| Pattern | Example |
|---------|---------|
| Binary task success | `1.0 if task.is_successful() else 0.0` |
| Shaped step reward | `base * exp(-decay * steps) + thinking_bonus` |
| Timeout (no termination) | `0.0` |
| Partial credit | `sum(subtask_rewards) / num_subtasks` |

Set on Sample: incremental `sample.reward = env.get_reward()`, history-based `for s in trajectory_samples: s.reward = reward`.

## Common Pitfalls

1. **Blocking `ray.get()`**: Always wrap with `asyncio.to_thread(ray.get, ...)` in async methods. #1 cause of poor throughput. See `examples/android_world/README.md` "Async Environment Calls".

2. **Forgetting to release workers**: Always `try/finally` in `generate()` calling `env.close()`. Lost workers are gone from the pool permanently.

3. **Observation tokens consuming budget**: Both model output AND observation tokens count against `max_context_len`. Keep observation templates minimal.

4. **BOS token duplication**: Strip leading BOS when encoding mid-sequence observations:
   ```python
   bos_id = state.tokenizer.bos_token_id
   if bos_id is not None and obs_ids and obs_ids[0] == bos_id:
       obs_ids = obs_ids[1:]
   ```

5. **Thread pool exhaustion**: `asyncio.to_thread()` default pool is `min(32, os.cpu_count() + 4)`. For >32 concurrent samples, increase it:
   ```python
   from concurrent.futures import ThreadPoolExecutor
   import asyncio
   asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=256))
   ```

6. **`loss_mask`/`rollout_log_probs` sizing**: Must have `len == response_length`. Assertion in `_convert_samples_to_train_data` checks this.

## Checklist

- [ ] `env_worker.py` — Worker with `__init__`, `reset`, `step`, `close`
- [ ] `env_pool.py` — Singleton pool with `get_instance`, `acquire`, `release`, `close`
- [ ] `env_your_env.py` — `BaseInteractionEnv` subclass with async `reset`/`step`
- [ ] `prompts.py` — System prompt + turn templates
- [ ] `rollout.py` — Incremental `generate()`
- [ ] `rollout_history.py` — (Optional) History-based `generate()`
- [ ] `config.yaml` — `num_workers`, `max_turns`, resources, env settings
- [ ] `data/tasks.jsonl` — Task dataset
- [ ] `run_your_env.sh` — Launch script
- [ ] `tests/` — Import, pool, and rollout tests
- [ ] All `ray.get()` calls wrapped in `asyncio.to_thread()`
- [ ] `env.close()` in `finally` block
- [ ] Observation tokens use `loss_mask=0` and `logprob=0.0`
- [ ] BOS token stripped from mid-sequence observations
