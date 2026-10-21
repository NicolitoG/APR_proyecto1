from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import meteogym
from meteogym import BatteryConfig, MeteoGymEnv, SimulationConfig
from meteogym.orbits import ContactSnapshot, OrbitSnapshot
from meteogym.workload import Workload


class StaticOrbit:
    satellite_ids = (1, 541)
    constellation_size = 1080
    default_start_time = "2016-10-29T19:52:11Z"

    def __init__(self):
        self.ranges = np.array([600.0, 800.0])
        self.visible = np.array([True, True])
        self.sunlit = np.array([False, False])

    def snapshot(self, time):
        return OrbitSnapshot(
            np.array([[6900.0, 0, 0], [0, 6900.0, 0]]),
            np.zeros((2, 3)),
            np.zeros((2, 3)),
            self.sunlit.copy(),
        )

    def contacts(self, time, users, min_elevation_deg=0):
        shape = (2, len(users))
        return ContactSnapshot(
            np.broadcast_to(self.visible[:, None], shape),
            np.broadcast_to(self.ranges[:, None], shape),
            np.full(shape, 60.0),
        )


def make_env(**kwargs):
    config = SimulationConfig(
        satellite_ids=(1, 541),
        num_users=1,
        duration_s=20,
        step_s=1,
        initial_tasks=True,
        task_rate_per_minute=(0, 0),
        task_size_mb=(1, 1),
        task_deadline_s=(10, 10),
        uplink_mbps=8,
    )
    return MeteoGymEnv(replace(config, **kwargs), orbit_model=StaticOrbit())


def test_nearest_active_satellite_completes_and_consumes_energy():
    env = make_env()
    observation, _ = env.reset(seed=3)
    initial_soc = observation["battery_soc"].copy()
    observation, reward, terminated, truncated, info = env.step([1, 1])
    assert reward == 1 and not terminated and not truncated
    assert info["task_events"][0]["satellite_id"] == 1
    np.testing.assert_allclose(info["task_energy_j"], [2, 0])
    assert observation["battery_soc"][0] < observation["battery_soc"][1] < initial_soc[1]
    assert info["generated_tasks"] == info["completed_tasks"] == 1


def test_passive_nearest_is_skipped():
    env = make_env()
    env.reset(seed=3)
    _, reward, _, _, info = env.step([0, 1])
    assert reward == 1
    assert info["task_events"][0]["satellite_id"] == 541


def test_deadline_fails_once_and_never_transmits_after_expiry():
    env = make_env(task_deadline_s=(0.5, 0.5))
    env.reset(seed=3)
    _, reward, _, _, info = env.step([1, 0])
    assert reward == -1
    assert info["task_events"][0]["time_s"] == 0.5
    assert info["task_energy_j"][0] == 1  # 0.5 MB before deadline
    assert env.step([1, 0])[1] == 0


def test_upload_completion_at_deadline_succeeds():
    env = make_env(task_deadline_s=(1, 1))
    env.reset(seed=3)
    assert env.step([1, 0])[1] == 1


def test_equal_bandwidth_sharing_and_single_pending_task():
    env = make_env(num_users=2)
    env.reset(seed=4)
    assert env.step([1, 0])[1] == 0
    assert all(task.remaining_mb == 0.5 for task in env.workload.tasks)
    assert env.workload.generated == 2
    assert env.step([1, 0])[1] == 2


def test_partial_upload_restarts_on_satellite_handover():
    env = make_env(uplink_mbps=4)
    env.reset(seed=3)
    assert env.step([1, 1])[1] == 0
    assert env.workload.tasks[0].remaining_mb == 0.5
    np.testing.assert_allclose(env._observation()["task_remaining_mb"][:, 0], [0.5, 1.0])
    env.orbits.ranges[:] = [900, 500]
    env._refresh_geometry()
    assert env.step([1, 1])[1] == 0
    assert env.workload.tasks[0].remaining_mb == 0.5
    np.testing.assert_allclose(env._observation()["task_remaining_mb"][:, 0], [1.0, 0.5])
    _, reward, _, _, info = env.step([1, 1])
    assert reward == 1
    assert info["task_events"][0]["satellite_id"] == 541


def test_hidden_users_mask_their_task_information():
    env = make_env()
    env.orbits.visible[:] = [False, True]
    obs, _ = env.reset(seed=3)
    assert obs["user_ids"][0, 0] == -1
    for field in (
        "distance_km",
        "task_present",
        "task_size_mb",
        "task_remaining_mb",
        "task_time_left_s",
    ):
        assert obs[field][0, 0] == 0
    _, reward, _, _, info = env.step([1, 0])
    assert reward == 0 and info["assignments"][0] == -1


def test_cutoff_overrides_actions_and_sunlight_recharges():
    env = make_env(initial_soc=0.2)
    env.orbits.sunlit[:] = True
    initial, _ = env.reset(seed=3)
    obs, reward, _, _, info = env.step([1, 1])
    assert reward == 0
    assert not info["effective_active"].any()
    assert (obs["battery_soc"] > initial["battery_soc"]).all()


def test_energy_budget_limits_work_without_fake_success():
    battery = BatteryConfig(
        capacity_ah=0.001, passive_current_a=0, active_current_a=0, solar_current_a=0
    )
    env = make_env(initial_soc=0.201, battery=battery)
    env.reset(seed=3)
    obs, reward, _, _, info = env.step([1, 0])
    assert reward == 0
    assert 0 < info["task_energy_j"][0] < 2
    assert 0 < env.workload.tasks[0].remaining_mb < 1
    assert obs["battery_soc"][0] >= 0.2 - 1e-12


def test_fractional_last_step_censors_pending_tasks():
    env = make_env(duration_s=1.5)
    env.reset(seed=3)
    assert not env.step([0, 0])[3]
    _, reward, terminated, truncated, info = env.step([0, 0])
    assert reward == 0 and not terminated and truncated
    assert info["step_s"] == 0.5 and info["elapsed_s"] == 1.5
    assert info["pending_tasks"] == 1 and info["failed_tasks"] == 0
    with pytest.raises(RuntimeError):
        env.step([0, 0])


def test_user_departure_fails_task_and_return_preserves_dimensions():
    env = make_env(user_departure_rate_per_minute=1e6, user_return_rate_per_minute=1e6)
    initial, _ = env.reset(seed=2)
    obs, reward, _, _, info = env.step([0, 0])
    assert reward == -1 and info["task_events"][0]["status"] == "user_departed"
    assert not obs["user_online"].any()
    obs, reward, _, _, _ = env.step([0, 0])
    assert obs["user_online"].all() and reward == 0
    assert obs.keys() == initial.keys()
    assert env.observation_space.contains(obs)


def test_randomized_workload_rates_and_gated_arrivals():
    config = SimulationConfig(num_users=1000, task_rate_per_minute=(1, 3), initial_tasks=True)
    work = Workload(config, np.random.default_rng(9))
    assert np.all((work.rates >= 1 / 60) & (work.rates <= 3 / 60))
    assert np.std(work.rates) > 0
    assert work.generate_due(1e6) == 0
    assert work.generated == 1000
    work.finish(0, 10)
    assert work.next_arrival_s[0] > 10
    assert work.generate_due(work.next_arrival_s[0]) == 1
    assert work.tasks[0].created_s > 10
    # Uniform area sampling is uniform in sin(latitude), not latitude.
    assert abs(np.sin(np.deg2rad(work.locations_deg[:, 0])).mean()) < 0.05


def test_config_round_trip_and_validation():
    config = SimulationConfig()
    assert SimulationConfig.from_dict(config.to_dict()) == config
    for changes in (
        {"step_s": 0},
        {"task_size_mb": (2, 1)},
        {"satellite_ids": (1, 1)},
        {"initial_soc": 2},
        {"user_locations_deg": ((0, 0),)},
        {"task_energy_j_per_mb": -1},
        {"duration_s": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(config, **changes)


def test_gymnasium_contract_and_deterministic_replay():
    env = make_env()
    check_env(env, skip_render_check=True)
    trajectories = []
    for _ in range(2):
        env.reset(seed=7)
        trajectories.append([env.step([1, 0]) for _ in range(3)])
    for first, second in zip(*trajectories):
        for key in first[0]:
            np.testing.assert_array_equal(first[0][key], second[0][key])
        assert first[1:4] == second[1:4]
    assert meteogym.MeteoGymEnv is MeteoGymEnv


def test_real_registered_environment_and_snapshot_ownership():
    env = gym.make("MeteoGym-v0", config=SimulationConfig(num_users=2, duration_s=10))
    obs, info = env.reset(seed=1)
    assert info["constellation_size"] == 1080
    saved_position = obs["position_km"].copy()
    new_obs, _, _, truncated, _ = env.step([1, 1])
    np.testing.assert_array_equal(obs["position_km"], saved_position)
    assert not np.array_equal(obs["position_km"], new_obs["position_km"])
    assert truncated
    assert env.observation_space.contains(new_obs)
    assert new_obs["time_s"].dtype == np.float64
    env.close()
