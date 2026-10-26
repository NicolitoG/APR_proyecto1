"""Arrival clocks and per-user queues, independent of orbital propagation."""

from dataclasses import replace

import numpy as np
import pytest

from meteogym import SimulationConfig
from meteogym.workload import Workload


def make_workload(**settings):
    config = SimulationConfig(
        num_users=1,
        user_locations_deg=((0.0, 0.0),),
        task_rate_per_minute=(60.0, 60.0),
        task_size_mb=(1.0, 1.0),
        task_deadline_s=(5.0, 5.0),
        task_arrival_mode="queued",
    )
    return Workload(replace(config, **settings), np.random.default_rng(7))


def test_queued_arrivals_catch_up_with_original_creation_times(monkeypatch):
    workload = make_workload()
    workload.next_arrival_s[:] = 0.25
    monkeypatch.setattr(workload, "_wait", lambda user: 0.5)

    assert workload.generate_due(1.3) == 3
    tasks = list(workload.queues[0])
    assert [task.created_s for task in tasks] == [0.25, 0.75, 1.25]
    assert [task.deadline_s for task in tasks] == [5.25, 5.75, 6.25]
    assert [task.task_id for task in tasks] == [0, 1, 2]
    assert workload.generated == workload.pending_count == 3
    assert workload.tasks == [tasks[0]]
    assert workload.next_arrival_s[0] == 1.75
    assert workload.generate_due(1.3) == 0


def test_queued_completion_preserves_arrival_clock_and_other_tasks(monkeypatch):
    workload = make_workload(initial_tasks=True)
    workload.next_arrival_s[:] = 0.5
    monkeypatch.setattr(workload, "_wait", lambda user: 1.0)
    workload.generate_due(2.0)
    first, second, third = workload.queues[0]

    assert workload.finish(0, 2.0, task=second) is second
    assert list(workload.queues[0]) == [first, third]
    assert workload.finish(0, 2.0) is first
    assert workload.tasks == [third]
    assert workload.next_arrival_s[0] == 2.5
    assert workload.finish(0, 2.0) is third
    assert workload.tasks == [None]
    assert workload.next_arrival_s[0] == 2.5
    assert workload.generate_due(2.5) == 1
    assert workload.tasks[0].created_s == 2.5


@pytest.mark.parametrize("initial_tasks, expected", [(False, 0), (True, 1)])
@pytest.mark.parametrize("mode", ["single", "queued"])
def test_zero_rate_only_allows_explicit_initial_task(initial_tasks, expected, mode):
    workload = make_workload(
        task_arrival_mode=mode,
        task_rate_per_minute=(0.0, 0.0),
        initial_tasks=initial_tasks,
    )
    assert workload.generated == workload.pending_count == expected
    assert workload.generate_due(1e6) == 0
    assert np.isinf(workload.next_arrival_s[0])
    if initial_tasks:
        assert workload.finish(0, 1e6).created_s == 0.0
        assert workload.generate_due(2e6) == 0


def test_initial_queued_task_keeps_normal_arrival_clock():
    workload = make_workload(initial_tasks=True)
    first_arrival = workload.next_arrival_s[0]
    assert workload.tasks[0].created_s == 0.0
    assert np.isfinite(first_arrival) and first_arrival > 0
    assert workload.generate_due(first_arrival) == 1
    assert workload.pending_count == 2
    assert workload.queues[0][-1].created_s == first_arrival


def test_departure_fails_every_task_and_offline_users_do_not_accumulate(monkeypatch):
    workload = make_workload(
        user_departure_rate_per_minute=1e6,
        user_return_rate_per_minute=1e6,
    )
    workload.next_arrival_s[:] = 0.25
    monkeypatch.setattr(workload, "_wait", lambda user: 0.5)
    workload.generate_due(1.3)
    pending = list(workload.queues[0])

    assert workload.update_presence(1.3, 1.3) == [(0, task) for task in pending]
    assert not workload.online[0]
    assert workload.pending_count == 0
    assert np.isinf(workload.next_arrival_s[0])
    assert workload.generate_due(100.0) == 0
    assert workload.update_presence(100.0, 1.0) == []
    assert workload.online[0]
    assert workload.next_arrival_s[0] == 100.5
    assert workload.generate_due(100.5) == 1
    assert workload.tasks[0].created_s == 100.5


def test_single_mode_keeps_boundary_admission_and_waits_until_finished(monkeypatch):
    workload = make_workload(task_arrival_mode="single")
    workload.next_arrival_s[:] = 0.25
    monkeypatch.setattr(workload, "_wait", lambda user: 0.5)

    assert workload.generate_due(1.3) == 1
    assert workload.tasks[0].created_s == 1.3
    assert workload.tasks[0].deadline_s == 6.3
    assert workload.generate_due(100.0) == 0
    assert workload.pending_count == 1
    workload.finish(0, 100.0)
    assert workload.next_arrival_s[0] == 100.5
    assert workload.generate_due(101.0) == 1
    assert workload.tasks[0].created_s == 101.0


def test_queued_workload_is_seed_reproducible():
    first = make_workload(task_size_mb=(0.1, 2.0), task_deadline_s=(1.0, 10.0))
    second = make_workload(task_size_mb=(0.1, 2.0), task_deadline_s=(1.0, 10.0))
    first.generate_due(20.0)
    second.generate_due(20.0)
    assert first.queues == second.queues
    np.testing.assert_array_equal(first.next_arrival_s, second.next_arrival_s)


def test_finishing_without_pending_task_is_rejected():
    workload = make_workload(task_rate_per_minute=(0.0, 0.0))
    with pytest.raises(RuntimeError, match="without a pending task"):
        workload.finish(0, 0.0)


@pytest.mark.parametrize("mode", ["single", "queued"])
def test_arrival_mode_configuration_round_trip(mode):
    config = SimulationConfig(task_arrival_mode=mode)
    assert SimulationConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("mode", ["queue", "", None, 1])
def test_invalid_arrival_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="task_arrival_mode"):
        SimulationConfig(task_arrival_mode=mode)


def test_single_arrival_mode_remains_the_default():
    assert SimulationConfig().task_arrival_mode == "single"
