"""Queued batches, per-step loss, voltage failure and recovery contracts."""

from dataclasses import replace

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from test_env import make_env

from meteogym import BatteryConfig
from meteogym.workload import Task


def queued_env(**kwargs):
    return make_env(task_arrival_mode="queued", initial_tasks=False, **kwargs)


def enqueue(env, user, size, deadline=10):
    task = Task(env.workload.generated, size, size, env.elapsed_s, deadline)
    env.workload.queues[user].append(task)
    env.workload.generated += 1
    return task


def assert_conserved(info):
    assert info["generated_tasks"] == (
        info["completed_tasks"] + info["failed_tasks"] + info["pending_tasks"]
    )
    assert info["total_reward"] == info["completed_tasks"] - info["failed_tasks"]


def test_drains_multiple_fifo_tasks_in_one_step():
    env = queued_env()
    env.reset(seed=1)
    tasks = [enqueue(env, 0, size) for size in (0.1, 0.2, 0.3)]
    obs, reward, _, _, info = env.step([1, 0])
    assert reward == 3
    assert [event["task_id"] for event in info["task_events"]] == [t.task_id for t in tasks]
    assert [event["time_s"] for event in info["task_events"]] == pytest.approx([0.1, 0.3, 0.6])
    assert info["task_energy_j"][0] == pytest.approx(1.2)
    assert not obs["task_queue_length"].any()
    assert env.observation_space.contains(obs)
    assert_conserved(info)


def test_unfinished_assigned_batch_is_lost_unassigned_backlog_stays():
    env = queued_env(num_users=2)
    env.reset(seed=1)
    # Only user zero is visible, although both users have a backlog.
    contacts = env.contacts
    visible = np.array([[True, False], [True, False]])
    env.contacts = replace(contacts, visible=visible)
    enqueue(env, 0, 0.25)
    unfinished = enqueue(env, 0, 2)
    waiting = enqueue(env, 0, 0.1)
    unassigned = enqueue(env, 1, 1)
    _, reward, _, _, info = env.step([1, 0])
    assert reward == -1
    assert info["completed_tasks"] == 1 and info["failed_tasks"] == 2
    assert [event["task_id"] for event in info["task_events"][1:]] == [
        unfinished.task_id, waiting.task_id
    ]
    assert all(event["status"] == "transfer_incomplete" for event in info["task_events"][1:])
    assert not env.workload.queues[0]
    assert list(env.workload.queues[1]) == [unassigned]
    assert info["task_energy_j"][0] == pytest.approx(2)
    assert_conserved(info)


def test_bandwidth_is_shared_between_users_not_tasks():
    env = queued_env(num_users=2)
    env.reset(seed=1)
    enqueue(env, 0, 0.25)
    enqueue(env, 0, 0.25)
    enqueue(env, 1, 0.5)
    _, reward, _, _, info = env.step([1, 0])
    assert reward == 3
    assert info["task_energy_j"][0] == pytest.approx(2)
    assert_conserved(info)


def test_nonhead_deadline_and_deadline_completion_boundary():
    env = queued_env()
    env.reset(seed=1)
    enqueue(env, 0, 1, deadline=1)
    queued = enqueue(env, 0, 0.1, deadline=0.25)
    _, reward, _, _, info = env.step([1, 0])
    assert reward == 0
    assert [(event["status"], event["time_s"]) for event in info["task_events"]] == [
        ("deadline", 0.25), ("completed", 1)
    ]
    assert info["task_events"][0]["task_id"] == queued.task_id
    assert_conserved(info)


def test_queued_arrivals_accumulate_and_include_final_interval():
    env = queued_env(duration_s=3, task_rate_per_minute=(600, 600))
    initial, _ = env.reset(seed=8)
    previous = 0
    for step in range(3):
        obs, _, _, truncated, info = env.step([0, 0])
        assert info["generated_tasks"] > previous
        previous = info["generated_tasks"]
        assert obs["task_queue_length"][0, 0] == info["pending_tasks"]
        assert obs["task_queue_remaining_mb"][0, 0] == info["pending_tasks"]
        assert env.observation_space.contains(obs)
        assert_conserved(info)
    assert truncated and not initial["task_queue_length"].any()


def test_expired_new_arrivals_are_counted_before_departure():
    env = queued_env(
        task_rate_per_minute=(6000, 6000), task_deadline_s=(0.1, 0.1),
        user_departure_rate_per_minute=1e6,
    )
    env.reset(seed=9)
    _, _, _, _, info = env.step([0, 0])
    assert info["generated_tasks"] > 1 and info["pending_tasks"] == 0
    assert {event["status"] for event in info["task_events"]} == {"deadline", "user_departed"}
    assert_conserved(info)


@pytest.mark.parametrize("resume_voltage, charging_steps", [(5.5, 2), (8.4, 4)])
def test_voltage_failure_keeps_completed_tasks_and_recovers_for_new_batch(
    resume_voltage, charging_steps,
):
    battery = BatteryConfig(
        voltage_empty_v=6.5, voltage_full_v=8.4,
        capacity_ah=0.001, active_current_a=0, passive_current_a=0,
        internal_resistance_ohm=0, minimum_voltage_v=7.4, resume_soc=0.7,
        resume_voltage_v=resume_voltage,
    )
    env = queued_env(initial_soc=0.5, battery=battery)
    env.reset(seed=1)
    completed = enqueue(env, 0, 0.1)
    unfinished = [enqueue(env, 0, 1), enqueue(env, 0, 1)]
    obs, reward, _, _, info = env.step([1, 0])
    assert reward == -1
    assert info["task_events"][0]["task_id"] == completed.task_id
    assert info["task_events"][0]["status"] == "completed"
    assert {event["task_id"] for event in info["task_events"][1:]} == {
        task.task_id for task in unfinished
    }
    assert {event["status"] for event in info["task_events"][1:]} == {"battery_cutoff"}
    assert 0 < info["active_duration_s"][0] < 1
    assert info["battery_failure_events"][0]["voltage_v"] == pytest.approx(7.4)
    assert obs["battery_suspended"][0] and not obs["available"][0]
    assert_conserved(info)
    new_task = enqueue(env, 0, 0.1)
    env.orbits.sunlit[:] = True
    env._refresh_geometry()
    for step in range(charging_steps):
        obs, reward, _, _, info = env.step([1, 0])
        assert reward == 0 and env.workload.tasks[0] is new_task
        assert not info["active_duration_s"][0]
        assert bool(obs["available"][0]) == (step == charging_steps - 1)
    assert env.battery.open_circuit_voltage_v()[0] >= resume_voltage
    obs, reward, _, _, info = env.step([1, 0])
    assert reward == 1 and not obs["battery_suspended"][0]
    assert [event["task_id"] for event in info["task_events"]] == [new_task.task_id]
    assert_conserved(info)


def test_high_voltage_cutoff_requires_actual_charge_before_recovery():
    battery = BatteryConfig(
        voltage_empty_v=6.5, voltage_full_v=8.4,
        capacity_ah=0.1, active_current_a=0, passive_current_a=0,
        minimum_voltage_v=7.44, resume_soc=0.3,
    )
    env = queued_env(initial_soc=0.5, battery=battery)
    env.reset(seed=1)
    enqueue(env, 0, 1)
    obs, _, _, _, info = env.step([1, 0])
    assert info["battery_failure_events"]
    assert obs["battery_suspended"][0]
    obs, _, _, _, _ = env.step([1, 0])
    assert obs["battery_suspended"][0]
    env.orbits.sunlit[:] = True
    env._refresh_geometry()
    obs, _, _, _, _ = env.step([1, 0])
    assert obs["available"][0]


def test_voltage_cutoff_does_not_transfer_more_than_bandwidth_allows():
    battery = BatteryConfig(
        voltage_empty_v=6.5, voltage_full_v=8.4,
        capacity_ah=0.001,
        active_current_a=0,
        passive_current_a=0,
        internal_resistance_ohm=0,
        minimum_voltage_v=7.4,
        resume_soc=0.7,
    )
    env = queued_env(initial_soc=0.5, battery=battery)
    env.reset(seed=1)
    enqueue(env, 0, 1)
    _, _, _, _, info = env.step([1, 0])
    assert info["battery_failure_events"]
    duration = info["active_duration_s"][0]
    assert 0 < duration < env.config.step_s
    bandwidth_energy_limit = (
        env.config.uplink_mbps / 8 * duration * env.config.task_energy_j_per_mb
    )
    assert info["task_energy_j"][0] <= bandwidth_energy_limit + 1e-10
    assert_conserved(info)


def test_free_processing_still_respects_bandwidth():
    env = queued_env(task_energy_j_per_mb=0)
    env.reset(seed=1)
    enqueue(env, 0, 1)
    enqueue(env, 0, 1)
    _, reward, _, _, info = env.step([1, 0])
    assert reward == 0
    assert info["completed_tasks"] == info["failed_tasks"] == 1
    assert not info["task_energy_j"].any()


def test_queued_gymnasium_contract_and_seeded_trajectories():
    env = queued_env(task_rate_per_minute=(30, 120))
    check_env(env, skip_render_check=True)
    trajectories = []
    for _ in range(2):
        env.reset(seed=19)
        trajectories.append([env.step([1, 0]) for _ in range(5)])
    for first, second in zip(*trajectories):
        for key in first[0]:
            np.testing.assert_array_equal(first[0][key], second[0][key])
        assert first[1:4] == second[1:4]
        assert first[4]["task_events"] == second[4]["task_events"]
        assert_conserved(first[4])
