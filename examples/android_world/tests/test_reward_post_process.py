"""Unit tests for post_process_rewards_history.

Verifies that GRPO reward normalization correctly handles variable-length
trajectories from rollout_history.py. No GPU, emulators, or Ray required.

Run from the slime repo root:
    pytest examples/android_world/tests/test_reward_post_process.py -v
"""

from __future__ import annotations

from argparse import Namespace
from typing import Any

import pytest

from slime.utils.types import Sample


def _make_args(**overrides) -> Namespace:
    defaults = dict(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        reward_key=None,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _make_sample(group_index: int, index: int, reward: float, **kwargs: Any) -> Sample:
    return Sample(
        group_index=group_index,
        index=index,
        reward=reward,
        tokens=[1, 2, 3],
        response_length=1,
        loss_mask=[1],
        status=Sample.Status.COMPLETED,
        **kwargs,
    )


def _import_func():
    from examples.android_world.rollout_history import post_process_rewards_history

    return post_process_rewards_history


class TestPostProcessRewardsHistory:
    """Tests for post_process_rewards_history."""

    def test_per_prompt_normalization_independent(self):
        """Different group_index groups are normalized independently."""
        func = _import_func()
        args = _make_args()

        # Prompt A (group_index=0): 2 trajectories, reward 0.0 and 1.0
        # Prompt B (group_index=1): 2 trajectories, reward 0.5 and 0.5
        samples = [
            _make_sample(group_index=0, index=0, reward=0.0),
            _make_sample(group_index=0, index=1, reward=1.0),
            _make_sample(group_index=1, index=0, reward=0.5),
            _make_sample(group_index=1, index=1, reward=0.5),
        ]

        raw, normalized = func(args, samples)

        # Raw rewards should be preserved
        assert raw == [0.0, 1.0, 0.5, 0.5]

        # Prompt A: mean=0.5, std=0.5 -> normalized = [-1.0, 1.0] (approx)
        assert normalized[0] < 0  # 0.0 is below mean
        assert normalized[1] > 0  # 1.0 is above mean

        # Prompt B: mean=0.5, std=0 -> (0.5-0.5)/(0+1e-6) ≈ 0
        assert abs(normalized[2]) < 1e-3
        assert abs(normalized[3]) < 1e-3

    def test_equal_trajectory_weight_regardless_of_length(self):
        """Each trajectory gets equal weight regardless of step count."""
        func = _import_func()
        args = _make_args()

        # One prompt (group_index=0) with 2 trajectories:
        # Trajectory 0: 5 steps, reward=1.0
        # Trajectory 1: 2 steps, reward=0.0
        samples = []
        for step in range(5):
            samples.append(_make_sample(group_index=0, index=0, reward=1.0))
        for step in range(2):
            samples.append(_make_sample(group_index=0, index=1, reward=0.0))

        raw, normalized = func(args, samples)

        # All steps in trajectory 0 should have the same normalized reward
        traj0_normalized = normalized[:5]
        assert all(v == traj0_normalized[0] for v in traj0_normalized)

        # All steps in trajectory 1 should have the same normalized reward
        traj1_normalized = normalized[5:]
        assert all(v == traj1_normalized[0] for v in traj1_normalized)

        # Trajectory 0 (reward=1.0) should be higher than trajectory 1 (reward=0.0)
        assert traj0_normalized[0] > traj1_normalized[0]

        # Normalization is based on 2 trajectory rewards [1.0, 0.0], not
        # 7 step rewards [1,1,1,1,1,0,0]. Mean=0.5, so traj0 positive, traj1 negative.
        assert traj0_normalized[0] > 0
        assert traj1_normalized[0] < 0

    def test_same_reward_all_steps_in_trajectory(self):
        """All steps in a trajectory get the same normalized reward."""
        func = _import_func()
        args = _make_args()

        # 2 prompts, each with 3 trajectories of varying length
        samples = []
        # Prompt 0
        for _ in range(3):  # traj 0, len=3
            samples.append(_make_sample(group_index=0, index=0, reward=1.0))
        for _ in range(1):  # traj 1, len=1
            samples.append(_make_sample(group_index=0, index=1, reward=0.5))
        for _ in range(4):  # traj 2, len=4
            samples.append(_make_sample(group_index=0, index=2, reward=0.0))
        # Prompt 1
        for _ in range(2):  # traj 0, len=2
            samples.append(_make_sample(group_index=1, index=0, reward=0.8))
        for _ in range(2):  # traj 1, len=2
            samples.append(_make_sample(group_index=1, index=1, reward=0.2))

        raw, normalized = func(args, samples)

        # Prompt 0, traj 0 (positions 0,1,2)
        assert normalized[0] == normalized[1] == normalized[2]
        # Prompt 0, traj 1 (position 3)
        # Prompt 0, traj 2 (positions 4,5,6,7)
        assert normalized[4] == normalized[5] == normalized[6] == normalized[7]

        # Prompt 1, traj 0 (positions 8,9)
        assert normalized[8] == normalized[9]
        # Prompt 1, traj 1 (positions 10,11)
        assert normalized[10] == normalized[11]

    def test_no_normalization_when_disabled(self):
        """When rewards_normalization is False, returns raw rewards unchanged."""
        func = _import_func()
        args = _make_args(rewards_normalization=False)

        samples = [
            _make_sample(group_index=0, index=0, reward=1.0),
            _make_sample(group_index=0, index=1, reward=2.0),
        ]

        raw, normalized = func(args, samples)
        assert raw == [1.0, 2.0]
        assert normalized == [1.0, 2.0]

    def test_no_normalization_for_non_grpo(self):
        """Non-GRPO advantage estimators skip normalization."""
        func = _import_func()
        args = _make_args(advantage_estimator="ppo")

        samples = [
            _make_sample(group_index=0, index=0, reward=1.0),
            _make_sample(group_index=0, index=1, reward=2.0),
        ]

        raw, normalized = func(args, samples)
        assert raw == [1.0, 2.0]
        assert normalized == [1.0, 2.0]

    def test_std_normalization_disabled(self):
        """With grpo_std_normalization=False, only mean-subtracts."""
        func = _import_func()
        args = _make_args(grpo_std_normalization=False)

        samples = [
            _make_sample(group_index=0, index=0, reward=3.0),
            _make_sample(group_index=0, index=1, reward=1.0),
        ]

        raw, normalized = func(args, samples)

        # Mean = 2.0, so normalized = [1.0, -1.0] (no std division)
        assert abs(normalized[0] - 1.0) < 1e-5
        assert abs(normalized[1] - (-1.0)) < 1e-5

    def test_single_trajectory_per_prompt(self):
        """Single trajectory per prompt results in zero after mean subtraction."""
        func = _import_func()
        args = _make_args(grpo_std_normalization=False)

        samples = [
            _make_sample(group_index=0, index=0, reward=5.0),
            _make_sample(group_index=0, index=0, reward=5.0),  # same traj, step 2
        ]

        raw, normalized = func(args, samples)

        # Only 1 trajectory, mean=5.0, so normalized=0.0
        assert abs(normalized[0]) < 1e-5
        assert abs(normalized[1]) < 1e-5

    def test_reinforce_plus_plus_baseline(self):
        """reinforce_plus_plus_baseline estimator also triggers normalization."""
        func = _import_func()
        args = _make_args(advantage_estimator="reinforce_plus_plus_baseline")

        samples = [
            _make_sample(group_index=0, index=0, reward=1.0),
            _make_sample(group_index=0, index=1, reward=0.0),
        ]

        raw, normalized = func(args, samples)
        # Should normalize (mean-subtract only, since rpp_baseline not in grpo/gspo for std)
        assert abs(normalized[0] - 0.5) < 1e-5
        assert abs(normalized[1] - (-0.5)) < 1e-5

    def test_interleaved_samples_from_multiple_prompts(self):
        """Handles samples that are interleaved (not grouped contiguously by prompt)."""
        func = _import_func()
        args = _make_args(grpo_std_normalization=False)

        # Interleave samples from 2 prompts
        samples = [
            _make_sample(group_index=0, index=0, reward=4.0),  # pos 0
            _make_sample(group_index=1, index=0, reward=10.0),  # pos 1
            _make_sample(group_index=0, index=1, reward=2.0),  # pos 2
            _make_sample(group_index=1, index=1, reward=6.0),  # pos 3
        ]

        raw, normalized = func(args, samples)

        # Prompt 0: rewards [4.0, 2.0], mean=3.0 -> [1.0, -1.0]
        assert abs(normalized[0] - 1.0) < 1e-5
        assert abs(normalized[2] - (-1.0)) < 1e-5

        # Prompt 1: rewards [10.0, 6.0], mean=8.0 -> [2.0, -2.0]
        assert abs(normalized[1] - 2.0) < 1e-5
        assert abs(normalized[3] - (-2.0)) < 1e-5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
