import os
import sys
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import traci
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------------
# Configuration / Hyperparams
# -------------------------
DIRECTORY_PATH = "results"
os.makedirs(DIRECTORY_PATH, exist_ok=True)

SIMULATION_STEPS = 10000
STEP_LENGTH = 0.1           # seconds per SUMO step
MIN_GREEN_SEC = 20         # minimum green in seconds
MIN_GREEN_STEPS = int(MIN_GREEN_SEC / STEP_LENGTH)

SWITCH_PENALTY = 3.0

ACTIONS = [0, 1]           # 0 = keep, 1 = switch
GAMMA = 0.9
EPSILON = 0.1              # fixed epsilon (simple baseline)
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
MIN_REPLAY_SIZE = 500
TARGET_UPDATE_FREQ = 500

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SUMO config - update to point to your multi_cross sumocfg
SUMO_CFG = "networks/multi_cross/multi_cross.sumocfg"
SUMO_COMMAND = ['sumo-gui', '-c', SUMO_CFG, '--step-length', str(STEP_LENGTH), '--start', '--quit-on-end']

# -------------------------
# SUMO startup
# -------------------------
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

print("Starting SUMO...")
traci.start(SUMO_COMMAND)
traci.gui.setSchema("View #0", "real world")

# -------------------------
# Small utilities & network
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_next, done = map(np.array, zip(*batch))
        return s, a, r, s_next, done

    def __len__(self):
        return len(self.buffer)

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=64):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def to_tensor(state_tuple):
    """Convert a 1-D state tuple/array to a (1, state_size) tensor on DEVICE."""
    arr = np.array(state_tuple, dtype=np.float32).reshape(-1)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

# -------------------------
# State / Reward helpers (per traffic light)
# -------------------------
def get_state(tl_id):
    """
    Compose a state for the traffic light `tl_id`.
    State = [queue from each lane-area detector (if present)]
            or halting counts from controlled lanes if no detectors,
            + current phase index (int)
    Returns tuple of ints/floats.
    """
    incoming_lanes = set(traci.trafficlight.getControlledLanes(tl_id))

    # use lane-area detectors if they exist and map to incoming lanes
    detectors = []
    try:
        all_detectors = traci.lanearea.getIDList()
        for det in all_detectors:
            try:
                lane_for_det = traci.lanearea.getLaneID(det)  # lane id associated with detector
            except Exception:
                lane_for_det = None
            if lane_for_det in incoming_lanes:
                detectors.append(det)
    except Exception:
        detectors = []

    detectors.sort()
    lane_queues = []

    if len(detectors) > 0:
        for det in detectors:
            # number of vehicles currently reported by detector
            q = traci.lanearea.getLastStepVehicleNumber(det)
            lane_queues.append(float(q))
    else:
        # fallback: use halting number per incoming lane
        incoming_lanes_list = sorted(list(incoming_lanes))
        for lane in incoming_lanes_list:
            q = traci.lane.getLastStepHaltingNumber(lane)
            lane_queues.append(float(q))

    current_phase = float(traci.trafficlight.getPhase(tl_id))
    # state tuple length is len(lane_queues) + 1
    return tuple(lane_queues + [current_phase])

def get_reward_from_state(state):
    """
    Reward is negative total queue length for that intersection.
    state is tuple where last element is phase index.
    """
    if len(state) <= 1:
        return -0.0
    total_queue = sum(state[:-1])
    return -float(total_queue)

def apply_action_for_tl(action, tl_id, current_step, last_switch_step_dict):
    """
    Apply action to a traffic light if action==1 and minimum green elapsed.
    Updates last_switch_step_dict[tl_id] when a switch occurs.
    """
    if action == 0:
        return False
    if current_step - last_switch_step_dict[tl_id] < MIN_GREEN_STEPS:
        return False

    # get number of phases for this TL
    program = traci.trafficlight.getAllProgramLogics(tl_id)[0]
    num_phases = len(program.phases)
    current_phase = traci.trafficlight.getPhase(tl_id)
    next_phase = (current_phase + 1) % num_phases
    traci.trafficlight.setPhase(tl_id, next_phase)
    last_switch_step_dict[tl_id] = current_step
    return True

# -------------------------
# Initialize multi-agent structures
# -------------------------
TRAFFIC_LIGHTS = traci.trafficlight.getIDList()
print("Detected traffic lights:", TRAFFIC_LIGHTS)

# last switch tracker per TL to enforce MIN_GREEN
last_switch_step = {tl: -MIN_GREEN_STEPS for tl in TRAFFIC_LIGHTS}

# Per-agent networks, buffers, optimizers
policy_nets = {}
target_nets = {}
optimizers = {}
replay_buffers = {}
state_sizes = {}
action_size = len(ACTIONS)

# Compute initial state sizes per TL (state shape can vary between TLs)
for tl in TRAFFIC_LIGHTS:
    # Step one simulation step to ensure detectors/lanes are available? There was an initial traci.start and GUI; usually detectors are ready.
    st = get_state(tl)
    s_dim = len(st)
    state_sizes[tl] = s_dim

    policy_nets[tl] = DQN(s_dim, action_size).to(DEVICE)
    target_nets[tl] = DQN(s_dim, action_size).to(DEVICE)
    target_nets[tl].load_state_dict(policy_nets[tl].state_dict())
    target_nets[tl].eval()

    optimizers[tl] = optim.Adam(policy_nets[tl].parameters(), lr=LR)
    replay_buffers[tl] = ReplayBuffer(REPLAY_CAPACITY)

print("Per-TL state sizes:", state_sizes)

# -------------------------
# Helper: select action for a given TL and state (epsilon-greedy)
# -------------------------
def select_action_for_tl(tl_id, state, epsilon=EPSILON):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    state_t = to_tensor(state)   # shape (1, s_dim)
    with torch.no_grad():
        qvals = policy_nets[tl_id](state_t)  # (1, action_size)
        act = int(torch.argmax(qvals, dim=1).item())
    return act

# -------------------------
# DQN update for a single TL (sample from its own buffer)
# -------------------------
def update_agent(tl_id):
    if len(replay_buffers[tl_id]) < MIN_REPLAY_SIZE:
        return None

    s_batch, a_batch, r_batch, s2_batch, d_batch = replay_buffers[tl_id].sample(BATCH_SIZE)

    # Convert to tensors. s_batch and s2_batch are arrays of tuples with fixed length = state_sizes[tl_id]
    states_t = torch.tensor(s_batch, dtype=torch.float32).to(DEVICE)       # (B, s)
    actions_t = torch.tensor(a_batch, dtype=torch.long).to(DEVICE)        # (B,)
    rewards_t = torch.tensor(r_batch, dtype=torch.float32).to(DEVICE)     # (B,)
    next_states_t = torch.tensor(s2_batch, dtype=torch.float32).to(DEVICE) # (B, s)
    dones_t = torch.tensor(d_batch.astype(np.float32), dtype=torch.float32).to(DEVICE)

    # Q(s,a)
    q_vals = policy_nets[tl_id](states_t)                      # (B, A)
    q_vals_a = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # target: r + gamma * max_a' Q_target(s', a') * (1 - done)
    with torch.no_grad():
        next_q = target_nets[tl_id](next_states_t)            # (B, A)
        next_q_max, _ = next_q.max(dim=1)                     # (B,)
        target_q = rewards_t + GAMMA * next_q_max * (1.0 - dones_t)

    loss = F.mse_loss(q_vals_a, target_q)

    optimizers[tl_id].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_nets[tl_id].parameters(), 10)
    optimizers[tl_id].step()

    return float(loss.item())

# -------------------------
# Main loop
# -------------------------
print("\n=== Starting multi-agent independent DQN ===")
system_steps = []
system_total_queue = []
system_total_wait = []
system_total_reward = []
cumulative_rewards_per_tl = {tl: 0.0 for tl in TRAFFIC_LIGHTS}
total_cumulative_reward = 0.0

for step in range(SIMULATION_STEPS):
    if traci.simulation.getMinExpectedNumber() <= 0:
        # no more vehicles expected
        break

    # 1) Observe states for each TL
    states = {tl: get_state(tl) for tl in TRAFFIC_LIGHTS}

    # 2) Choose actions (epsilon-greedy) independently
    actions = {tl: select_action_for_tl(tl, states[tl]) for tl in TRAFFIC_LIGHTS}

    # 3) Apply actions (subject to MIN_GREEN constraint)
    switched_flags = {}
    for tl in TRAFFIC_LIGHTS:
        switched_flags[tl] = apply_action_for_tl(actions[tl], tl, step, last_switch_step)

    # 4) Advance SUMO one step
    traci.simulationStep()

    # 5) Observe new states and compute rewards
    next_states = {tl: get_state(tl) for tl in TRAFFIC_LIGHTS}
    rewards = {tl: get_reward_from_state(next_states[tl]) for tl in TRAFFIC_LIGHTS}

    # Apply switching penalty
    for tl in TRAFFIC_LIGHTS:
        reward = get_reward_from_state(next_states[tl])
        # Apply switching penalty if TL switched
        if switched_flags[tl]:
            reward -= SWITCH_PENALTY
        rewards[tl] = reward


    done = False

    # 6) Store transitions in per-agent replay buffers
    for tl in TRAFFIC_LIGHTS:
        replay_buffers[tl].push(states[tl], actions[tl], rewards[tl], next_states[tl], done)
        cumulative_rewards_per_tl[tl] += rewards[tl]
        total_cumulative_reward += rewards[tl]

    # 7) Train each agent (independently) once per step (if buffer large enough)
    losses = {}
    for tl in TRAFFIC_LIGHTS:
        loss_val = update_agent(tl)
        losses[tl] = loss_val

    # 8) Sync target networks periodically
    if step % TARGET_UPDATE_FREQ == 0 and step > 0:
        for tl in TRAFFIC_LIGHTS:
            target_nets[tl].load_state_dict(policy_nets[tl].state_dict())

    # 9) Logging & system metrics
    total_queue = 0
    for tl in TRAFFIC_LIGHTS:
        # Use controlled lanes to accumulate queue (halting vehicles)
        lanes = traci.trafficlight.getControlledLanes(tl)
        total_queue += sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)

    total_wait = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())

    system_steps.append(step)
    system_total_queue.append(total_queue)
    system_total_wait.append(total_wait)
    system_total_reward.append(total_cumulative_reward)

    if step % 100 == 0:
        avg_loss = np.mean([v for v in losses.values() if v is not None]) if any(v is not None for v in losses.values()) else None
        qvals_debug = {}
        for tl in TRAFFIC_LIGHTS:
            try:
                qvals_debug[tl] = policy_nets[tl](to_tensor(get_state(tl))).detach().cpu().numpy()[0]
            except Exception:
                qvals_debug[tl] = None
        print(f"Step {step} | TotalQueue={total_queue} | TotalWait={total_wait:.1f} | CumReward={total_cumulative_reward:.1f} | AvgLoss={avg_loss}")
        print("  example Q-values (per TL):")
        for tl, qv in qvals_debug.items():
            print(f"    {tl}: {qv}")
        print("  last_switch_steps:", {tl: last_switch_step[tl] for tl in TRAFFIC_LIGHTS})

# -------------------------
# Shutdown SUMO
# -------------------------
traci.close()
print("SUMO closed. Training finished.")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,6))
plt.plot(system_steps, system_total_queue, label="System total queue")
plt.xlabel("Step")
plt.ylabel("Total queue (halting vehicles)")
plt.title("System total queue over time")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(DIRECTORY_PATH, "multi_dqn_total_queue.png"))
plt.show()

plt.figure(figsize=(10,6))
plt.plot(system_steps, system_total_wait, label="System total waiting time")
plt.xlabel("Step")
plt.ylabel("Total waiting time (s)")
plt.title("System total waiting time over time")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(DIRECTORY_PATH, "multi_dqn_total_wait.png"))
plt.show()

plt.figure(figsize=(10,6))
plt.plot(system_steps, system_total_reward, label="Cumulative reward")
plt.xlabel("Step")
plt.ylabel("Cumulative reward")
plt.title("Cumulative reward (sum across agents)")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(DIRECTORY_PATH, "multi_dqn_cum_reward.png"))
plt.show()