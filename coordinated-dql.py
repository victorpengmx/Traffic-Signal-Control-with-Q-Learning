"""Centralized/coordinated Deep Q-Learning for multi-intersection control.

This variant trains a single coordinator that observes the concatenated
state of every traffic light and chooses a joint action (0=keep, 1=switch)
for each intersection simultaneously. This enables the agent to reason
about cross-intersection dependencies instead of treating each controller
independently as in `multiagent-dql.py`.
"""

import collections
import itertools
import json
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import traci


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIRECTORY_PATH = "results"
os.makedirs(DIRECTORY_PATH, exist_ok=True)

SIMULATION_STEPS = 10_000
STEP_LENGTH = 0.1
MIN_GREEN_SEC = 20
MIN_GREEN_STEPS = int(MIN_GREEN_SEC / STEP_LENGTH)

ACTIONS = [0, 1]  # 0 = keep current phase, 1 = advance to next phase
SWITCH_PENALTY = 3.0

# DQN hyper-parameters (mirrors multiagent-dql baseline)
GAMMA = 0.9
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 10_000
MIN_REPLAY_SIZE = 500
TARGET_UPDATE_FREQ = 500

EPSILON = 0.1  # constant epsilon, consistent with cooperative/multi-agent scripts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# SUMO bootstrap
# ---------------------------------------------------------------------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo-gui")
SUMO_CFG = "networks/multi_cross/multi_cross.sumocfg"
SUMO_COMMAND = [
    SUMO_BINARY,
    "-c",
    SUMO_CFG,
    "--step-length",
    str(STEP_LENGTH),
    "--start",
    "--quit-on-end",
]

print("Starting SUMO for coordinated DQL...")
traci.start(SUMO_COMMAND)
if SUMO_BINARY.endswith("gui"):
    traci.gui.setSchema("View #0", "real world")


# ---------------------------------------------------------------------------
# Replay buffer and neural network
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int):
        self._buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self._buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int):
        batch = random.sample(self._buffer, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return s, a, r, s_next, done

    def __len__(self):
        return len(self._buffer)


class CoordinatorDQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        hidden = 256
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------
def get_lane_queues_for_tl(tl_id: str) -> List[float]:
    incoming_lanes = traci.trafficlight.getControlledLanes(tl_id)
    detector_ids = []
    for lane in incoming_lanes:
        detector_ids.extend(_LANE_DETECTORS.get(lane, []))

    if detector_ids:
        return [traci.lanearea.getLastStepVehicleNumber(det) for det in sorted(detector_ids)]

    return [traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes]


def get_state(tl_id: str) -> Tuple[float, ...]:
    lane_queues = get_lane_queues_for_tl(tl_id)
    current_phase = traci.trafficlight.getPhase(tl_id)
    return tuple(lane_queues + [float(current_phase)])


def pad_state(state: Sequence[float], pad_to: int) -> List[float]:
    padded = list(state)
    if len(padded) < pad_to:
        padded.extend([0.0] * (pad_to - len(padded)))
    return padded


def get_global_state() -> np.ndarray:
    pieces: List[float] = []
    for tl in TRAFFIC_LIGHTS:
        st = pad_state(get_state(tl), _MAX_STATE_LEN)
        pieces.extend(st)
    return np.asarray(pieces, dtype=np.float32)


def apply_joint_action(action_tuple: Tuple[int, ...], current_step: int) -> Dict[str, bool]:
    switched = {}
    for idx, tl in enumerate(TRAFFIC_LIGHTS):
        act = action_tuple[idx]
        switched_flag = False
        if act == 1 and current_step - _last_switch_step[tl] >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tl)[0]
            num_phases = len(program.phases)
            next_phase = (traci.trafficlight.getPhase(tl) + 1) % num_phases
            traci.trafficlight.setPhase(tl, next_phase)
            _last_switch_step[tl] = current_step
            switched_flag = True
        switched[tl] = switched_flag
    return switched


def measure_system_queue() -> float:
    total = 0.0
    for tl in TRAFFIC_LIGHTS:
        lanes = traci.trafficlight.getControlledLanes(tl)
        total += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
    return total


def measure_system_waiting() -> float:
    vehicle_ids = traci.vehicle.getIDList()
    return sum(traci.vehicle.getWaitingTime(v) for v in vehicle_ids)


def save_metric_history(tag: str, metric: str, steps, values):
    path = os.path.join(DIRECTORY_PATH, f"{tag}-{metric}.json")
    with open(path, "w") as fh:
        json.dump({"steps": steps, "values": values}, fh)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

TRAFFIC_LIGHTS = traci.trafficlight.getIDList()
if not TRAFFIC_LIGHTS:
    traci.close()
    raise RuntimeError("No traffic lights detected in the loaded SUMO network.")

print(f"Detected traffic lights: {TRAFFIC_LIGHTS}")

# Cache lane->detectors to avoid repeated lookups
_LANE_DETECTORS: Dict[str, List[str]] = {}
for det in traci.lanearea.getIDList():
    try:
        lane = traci.lanearea.getLaneID(det)
    except traci.TraCIException:
        continue
    _LANE_DETECTORS.setdefault(lane, []).append(det)

# Determine padding size (max state length over TLs)
_MAX_STATE_LEN = 0
for tl in TRAFFIC_LIGHTS:
    st_len = len(get_state(tl))
    _MAX_STATE_LEN = max(_MAX_STATE_LEN, st_len)

GLOBAL_STATE_DIM = _MAX_STATE_LEN * len(TRAFFIC_LIGHTS)
print(f"Global state dimension: {GLOBAL_STATE_DIM} (per TL max={_MAX_STATE_LEN})")

# Enumerate all joint action combinations (2^N for binary actions)
JOINT_ACTIONS: List[Tuple[int, ...]] = list(itertools.product(ACTIONS, repeat=len(TRAFFIC_LIGHTS)))
print(f"Joint action space size: {len(JOINT_ACTIONS)}")

_last_switch_step = {tl: -MIN_GREEN_STEPS for tl in TRAFFIC_LIGHTS}

replay_buffer = ReplayBuffer(REPLAY_CAPACITY)
policy_net = CoordinatorDQN(GLOBAL_STATE_DIM, len(JOINT_ACTIONS)).to(DEVICE)
target_net = CoordinatorDQN(GLOBAL_STATE_DIM, len(JOINT_ACTIONS)).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=LR)


def select_joint_action(state_vec: np.ndarray) -> int:
    if random.random() < EPSILON:
        return random.randrange(len(JOINT_ACTIONS))
    state_t = torch.from_numpy(state_vec).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        qvals = policy_net(state_t)
    return int(torch.argmax(qvals, dim=1).item())


def update_coordinator():
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffer.sample(BATCH_SIZE)
    states_t = torch.tensor(s_batch, dtype=torch.float32).to(DEVICE)
    actions_t = torch.tensor(a_batch, dtype=torch.long).to(DEVICE)
    rewards_t = torch.tensor(r_batch, dtype=torch.float32).to(DEVICE)
    next_states_t = torch.tensor(s2_batch, dtype=torch.float32).to(DEVICE)
    dones_t = torch.tensor(d_batch.astype(np.float32), dtype=torch.float32).to(DEVICE)

    q_vals = policy_net(states_t)
    q_vals_a = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states_t)
        next_q_max, _ = next_q.max(dim=1)
        target_q = rewards_t + GAMMA * next_q_max * (1.0 - dones_t)

    loss = F.mse_loss(q_vals_a, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 5.0)
    optimizer.step()

    return float(loss.item())


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
system_steps: List[int] = []
system_total_queue: List[float] = []
system_total_wait: List[float] = []
system_total_reward: List[float] = []

cumulative_reward = 0.0

print("\n=== Coordinated DQL training ===")

for step in range(SIMULATION_STEPS):
    if traci.simulation.getMinExpectedNumber() <= 0:
        break

    state_vec = get_global_state()
    action_idx = select_joint_action(state_vec)
    action_tuple = JOINT_ACTIONS[action_idx]

    switched_dict = apply_joint_action(action_tuple, step)
    traci.simulationStep()

    next_state_vec = get_global_state()
    total_queue = measure_system_queue()
    total_wait = measure_system_waiting()

    switch_penalty = SWITCH_PENALTY * sum(1 for flag in switched_dict.values() if flag)
    reward = -total_queue - switch_penalty
    cumulative_reward += reward

    done = False
    replay_buffer.push(state_vec, action_idx, reward, next_state_vec, done)

    loss_val = update_coordinator()

    if step % TARGET_UPDATE_FREQ == 0 and step > 0:
        target_net.load_state_dict(policy_net.state_dict())

    system_steps.append(step)
    system_total_queue.append(total_queue)
    system_total_wait.append(total_wait)
    system_total_reward.append(cumulative_reward)

    if step % 100 == 0:
        print(
            f"Step {step} | Queue={total_queue:.1f} | Wait={total_wait:.1f} | "
            f"Reward={cumulative_reward:.1f} | Eps={EPSILON:.3f} | Loss={loss_val}"
        )


# ---------------------------------------------------------------------------
# Shutdown & persistence
# ---------------------------------------------------------------------------
traci.close()
print("SUMO closed. Coordinated DQL training finished.")

save_metric_history("coordinated-dql", "waiting", system_steps, system_total_wait)
save_metric_history("coordinated-dql", "queue", system_steps, system_total_queue)


# ---------------------------------------------------------------------------
# Plotting helpers for quick inspection
# ---------------------------------------------------------------------------
def plot_metric(series, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(system_steps, series)
    plt.xlabel("Simulation step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    output_path = os.path.join(DIRECTORY_PATH, filename)
    plt.savefig(output_path)
    print(f"Saved plot to {output_path}")
    plt.close()


plot_metric(system_total_queue, "Total queue (vehicles)", "Coordinated DQL - Queue", "coordinated_dql_queue.png")
plot_metric(system_total_wait, "Total waiting time (s)", "Coordinated DQL - Waiting", "coordinated_dql_wait.png")
plot_metric(system_total_reward, "Cumulative reward", "Coordinated DQL - Reward", "coordinated_dql_reward.png")
