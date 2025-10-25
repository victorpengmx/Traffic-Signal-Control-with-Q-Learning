import os
import sys
import traci
import random
import numpy as np
import matplotlib.pyplot as plt

# SUMO environment setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration (GUI enabled, runs automatically)
sumo_config = [
    'sumo-gui', '-c', 'networks/simple_cross.sumocfg',
    '--step-length', '0.10', '--start', '--quit-on-end'
]

# Start SUMO and Traci automatically
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Parameters
SIMULATION_STEPS = 10000
TRAFFIC_LIGHT = "cluster_J1_J2_J4_J6"
STEP_LENGTH = 0.1
ACTIONS = [0, 1]  # 0 = keep phase, 1 = switch phase
ALPHA = 0.1       # Learning rate
GAMMA = 0.9       # Discount factor
EPSILON = 0.1     # Exploration rate

# Data
step_history = []
queue_history = []
waiting_time_history = []

# Q-table (state = simplified traffic condition, action = phase change or not)
Q = {}

def get_queue_length():
    # Total number of halted vehicles (queue length)
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
    # Define state as discretized queue length (rounded to nearest 5 vehicles)
    q_len = get_queue_length()
    return int(round(q_len / 5.0) * 5)

def choose_action(state):
    # Epsilon-greedy policy
    if random.uniform(0, 1) < EPSILON or state not in Q:
        return random.choice(ACTIONS)
    else:
        return np.argmax(Q[state])

def update_q_value(state, action, reward, next_state):
    # Initialize Q-values if not present
    if state not in Q:
        Q[state] = np.zeros(len(ACTIONS))
    if next_state not in Q:
        Q[next_state] = np.zeros(len(ACTIONS))

    # Q-learning update rule
    Q[state][action] = Q[state][action] + ALPHA * (
        reward + GAMMA * np.max(Q[next_state]) - Q[state][action]
    )

# Simulation Loop
print("\n=== Running Q-Learning Traffic Signal Control ===")
phase_step_counter = 0
current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT)

for step in range(SIMULATION_STEPS + 1):
    # Automatically stop if no vehicles left
    if traci.simulation.getMinExpectedNumber() <= 0:
        print("No more vehicles — stopping simulation.")
        break

    traci.simulationStep()
    state = get_state()

    # Choose action
    action = choose_action(state)

    # Apply action (0 = keep phase, 1 = switch phase)
    if action == 1:
        current_phase = (current_phase + 1) % len(
            traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT)[0].phases
        )
        traci.trafficlight.setPhase(TRAFFIC_LIGHT, current_phase)
        phase_step_counter = 0
    else:
        phase_step_counter += 1

    # Reward based on total waiting time (lower waiting = better)
    reward = -get_total_waiting_time()

    # Next state and Q-value update
    next_state = get_state()
    update_q_value(state, action, reward, next_state)

    # Record metrics every 100 steps
    if step % 100 == 0:
        total_q = get_queue_length()
        total_w = get_total_waiting_time()
        print(f"Step {step}: Queue={total_q}, WaitingTime={total_w:.2f}")

        step_history.append(step)
        queue_history.append(total_q)
        waiting_time_history.append(total_w)

# Clean shutdown
traci.close(False)
print("\nSimulation complete. SUMO closed automatically.")

# Plot Queue Length
plt.figure(figsize=(8, 6))
plt.plot(step_history, queue_history, marker='o', label="Total Queue Length")
plt.title("Queue Length over Steps (Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.show()

# Plot Waiting Time
plt.figure(figsize=(8, 6))
plt.plot(step_history, waiting_time_history, marker='o', label="Total Waiting Time")
plt.title("Total Waiting Time over Steps (Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.show()
