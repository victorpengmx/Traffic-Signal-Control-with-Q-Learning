import os
import sys
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import traci

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

DIRECTORY_PATH = "results"
os.makedirs(DIRECTORY_PATH, exist_ok=True)

# SUMO environment setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration (GUI enabled, runs automatically)
sumo_config = [
    'sumo-gui', '-c', 'networks/simple_cross_lad.sumocfg',
    '--step-length', '0.10', '--start', '--quit-on-end'
]

# Start SUMO and Traci automatically
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Hyperparameters
SIMULATION_STEPS = 10000
GAMMA = 0.9
EPSILON = 0.1                # fixed exploration rate
LR = 1e-3
BATCH_SIZE = 64
REPLAY_CAPACITY = 10000
MIN_REPLAY_SIZE = 500        # start training after this many transitions
TARGET_UPDATE_FREQ = 500     # steps between copying online -> target network
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Traffic / RL settings
ACTIONS = [0, 1]             # 0 = keep phase, 1 = switch phase

MIN_GREEN_SEC = 20        # real seconds
STEP_LENGTH = 0.1          # from your sumo-gui arguments
MIN_GREEN_STEPS = int(MIN_GREEN_SEC / STEP_LENGTH)

last_switch_step = -MIN_GREEN_STEPS

TRAFFIC_LIGHT = "cluster_J1_J2_J4_J6"

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

# DQN model
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
    arr = np.array(state_tuple, dtype=np.float32).reshape(-1)  # shape (state_size,)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)      # shape (1, state_size)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def get_lane_queue_length(lane_id):
    return traci.lanearea.getLastStepVehicleNumber(lane_id)

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)

def get_total_queue_length():
    total = 0
    for lane_id in traci.lane.getIDList():
        total += traci.lane.getLastStepHaltingNumber(lane_id)
    return total

def get_total_waiting_time():
    # Total waiting time of all vehicles
    vehicle_ids = traci.vehicle.getIDList()
    total = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    return total

def get_state():
    """
    State = tuple of queue lengths from lane-area detectors for incoming lanes
             + current traffic light phase.
    """
    incoming_lanes = set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT))

    detectors = [
        det for det in traci.lanearea.getIDList()
        if traci.lanearea.getLaneID(det) in incoming_lanes
    ]
    detectors.sort()

    # Collect queue length from each detector
    lane_queues = []
    for det in detectors:
        q = traci.lanearea.getLastStepVehicleNumber(det)
        q_discrete = int(round(q / 1.0))
        lane_queues.append(q_discrete)

    # If no detectors exist, use halting number
    if len(detectors) == 0:
        incoming_lanes = sorted(list(incoming_lanes))
        for lane in incoming_lanes:
            q = traci.lane.getLastStepHaltingNumber(lane)
            lane_queues.append(q)

    current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT)
    state = tuple(lane_queues + [current_phase])
    return state

def get_reward(state):
    total_queue = sum(state[:-1])  # exclude current_phase
    return -float(total_queue)

def apply_action(action, tls_id=TRAFFIC_LIGHT, current_simulation_step=None):
    global last_switch_step
    if action == 0:
        return
    elif action == 1:
        if current_simulation_step is None:
            return
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            last_switch_step = current_simulation_step

# Instantiate nets, optimizer, buffer
state_size = len(get_state())
action_size = len(ACTIONS)
policy_net = DQN(state_size, action_size).to(DEVICE)
target_net = DQN(state_size, action_size).to(DEVICE)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
loss_fn = nn.MSELoss()
replay_buffer = ReplayBuffer(REPLAY_CAPACITY)

# Epsilon-greedy & training step
def select_action(state, epsilon=EPSILON):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    else:
        state_t = to_tensor(state)  # (1, state_size)
        with torch.no_grad():
            qvals = policy_net(state_t)  # (1, action_size)
            action = int(torch.argmax(qvals, dim=1).item())
        return action

def update_dqn(batch_size=BATCH_SIZE):
    if len(replay_buffer) < batch_size:
        return None

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    # Convert to tensors
    states_t = torch.tensor(states, dtype=torch.float32).to(DEVICE)            # (B, S)
    actions_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)             # (B,)
    rewards_t = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)         # (B,)
    next_states_t = torch.tensor(next_states, dtype=torch.float32).to(DEVICE) # (B, S)
    dones_t = torch.tensor(dones.astype(np.float32), dtype=torch.float32).to(DEVICE)  # (B,)

    # Current Q-values (policy_net) for the actions taken
    q_values = policy_net(states_t)                      # (B, A)
    q_values_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)

    # Next-state values (target_net)
    with torch.no_grad():
        next_q_values = target_net(next_states_t)       # (B, A)
        next_q_max, _ = next_q_values.max(dim=1)        # (B,)
        target_q = rewards_t + GAMMA * next_q_max * (1.0 - dones_t)

    # Loss and optimization
    loss = loss_fn(q_values_a, target_q)
    optimizer.zero_grad()
    loss.backward()
    # gradient clipping optional
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10)
    optimizer.step()
    return loss.item()


# Main loop (online interaction + training)
step_history = []
reward_history = []
queue_history = []
waiting_time_history = []
cumulative_reward = 0.0

print("\n=== Starting DQN training (PyTorch) ===")

for step in range(SIMULATION_STEPS):
    current_simulation_step = step

    state = get_state()
    action = select_action(state, EPSILON)
    apply_action(action, tls_id=TRAFFIC_LIGHT, current_simulation_step=current_simulation_step)

    traci.simulationStep()

    new_state = get_state()
    reward = get_reward(new_state)
    done = False  # in continuous SUMO simulation, treat as non-terminal; could add episode logic if you wish

    replay_buffer.push(state, action, reward, new_state, done)
    cumulative_reward += reward

    # Train step if buffer big enough
    if len(replay_buffer) >= MIN_REPLAY_SIZE:
        loss_val = update_dqn(BATCH_SIZE)
    else:
        loss_val = None

    # Periodically update target network
    if step % TARGET_UPDATE_FREQ == 0 and step > 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Logging
    if step % 100 == 0:
        total_w = get_total_waiting_time()

        qvals_now = policy_net(to_tensor(state)).detach().cpu().numpy()[0]
        print(f"Step {step}, State: {state}, Action: {action}, NewState: {new_state}, Reward: {reward:.2f}, CumReward: {cumulative_reward:.2f}, Qvals: {qvals_now}, Loss: {loss_val} \n")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))
        waiting_time_history.append(total_w)

traci.close()
print("Training completed. Policy net:")
print(policy_net)

# Plot cumulative reward
plt.figure(figsize=(8,6))
plt.plot(step_history, reward_history, label="Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid(True)

# Plot Queue Length
plt.figure(figsize=(8, 6))
plt.plot(step_history, queue_history, marker='o', label="Total Queue Length")
plt.title("Queue Length over Steps (Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/dql-queuelength.png")
plt.show()

# Plot Waiting Time
plt.figure(figsize=(8, 6))
plt.plot(step_history, waiting_time_history, marker='o', label="Total Waiting Time")
plt.title("Total Waiting Time over Steps (Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/dql-waitingtime.png")
plt.show()

