import os
import sys
import traci
import random
import numpy as np
import matplotlib.pyplot as plt

DIRECTORY_PATH = "results"
os.makedirs(DIRECTORY_PATH, exist_ok=True)

# SUMO environment setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
sumo_config = [
    'sumo-gui', '-c', 'networks/multi_cross.sumocfg',
    '--step-length', '0.10', '--start', '--quit-on-end'
]

# Start SUMO and Traci
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Parameters
SIMULATION_STEPS = 10000
STEP_LENGTH = 0.1
ACTIONS = [0, 1]  # 0 = keep phase, 1 = switch phase
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1

# Hard constraint: minimum green time
MIN_GREEN_TIME = 25
MIN_STEPS_PER_PHASE = int(MIN_GREEN_TIME / STEP_LENGTH)

traffic_lights = traci.trafficlight.getIDList()
print(f"Detected {len(traffic_lights)} traffic lights: {traffic_lights}")

# Data
step_history = []
avg_queue_history = []
avg_waiting_time_history = []

phase_step_counter = {tl: 0 for tl in traffic_lights}

# Initialize Q-tables per intersection
Q_tables = {tl: {} for tl in traffic_lights}

# Track current phase per traffic light
current_phases = {tl: traci.trafficlight.getPhase(tl) for tl in traffic_lights}

def get_queue_length(tl):
    # Return total number of halted vehicles on incoming lanes of traffic light
    incoming_lanes = traci.trafficlight.getControlledLanes(tl)
    total = sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)
    return total

def get_total_waiting_time(tl):
    # Return total waiting time for vehicles approaching traffic light
    incoming_lanes = traci.trafficlight.getControlledLanes(tl)
    veh_ids = set()
    for lane in incoming_lanes:
        veh_ids.update(traci.lane.getLastStepVehicleIDs(lane))
    return sum(traci.vehicle.getWaitingTime(v) for v in veh_ids)

def get_state(tl):
    # Discretized queue length (rounded to nearest 5 vehicles)
    q_len = get_queue_length(tl)
    return int(round(q_len / 5.0) * 5)

def choose_action(tl, state):
    # Epsilon-greedy policy per intersection
    if random.uniform(0, 1) < EPSILON or state not in Q_tables[tl]:
        return random.choice(ACTIONS)
    else:
        return np.argmax(Q_tables[tl][state])

def update_q_value(tl, state, action, reward, next_state):
    # Independent Q-learning update for each intersection
    if state not in Q_tables[tl]:
        Q_tables[tl][state] = np.zeros(len(ACTIONS))
    if next_state not in Q_tables[tl]:
        Q_tables[tl][next_state] = np.zeros(len(ACTIONS))

    # Q-learning update rule
    Q_tables[tl][state][action] = Q_tables[tl][state][action] + ALPHA * (
        reward + GAMMA * np.max(Q_tables[tl][next_state]) - Q_tables[tl][state][action]
    )

# Simulation loop
print("\n=== Running Multi-Intersection Q-Learning Traffic Control ===")

for step in range(SIMULATION_STEPS + 1):
    # Automatically stop if no vehicles left
    if traci.simulation.getMinExpectedNumber() <= 0:
        print("No more vehicles — stopping simulation.")
        break

    traci.simulationStep()

    total_queue = 0
    total_waiting = 0

    # Loop through each intersection
    for tl in traffic_lights:
        state = get_state(tl)
        action = choose_action(tl, state)

        # Apply action (0 = keep phase, 1 = switch phase)
        if action == 1 and phase_step_counter[tl] >= MIN_STEPS_PER_PHASE:
            current_phases[tl] = (current_phases[tl] + 1) % len(
                traci.trafficlight.getCompleteRedYellowGreenDefinition(tl)[0].phases
            )
            traci.trafficlight.setPhase(tl, current_phases[tl])

        # Reward (negative waiting time)
        reward = -get_total_waiting_time(tl)

        next_state = get_state(tl)
        update_q_value(tl, state, action, reward, next_state)

        total_queue += get_queue_length(tl)
        total_waiting += get_total_waiting_time(tl)

    # Record every 100 steps
    if step % 100 == 0:
        avg_q = total_queue / len(traffic_lights)
        avg_w = total_waiting / len(traffic_lights)
        print(f"Step {step}: AvgQueue={avg_q:.2f}, AvgWait={avg_w:.2f}")

        step_history.append(step)
        avg_queue_history.append(avg_q)
        avg_waiting_time_history.append(avg_w)

# Clean shutdown
traci.close(False)
print("\nSimulation complete. SUMO closed automatically.")

# Plot queue length
plt.figure(figsize=(8, 6))
plt.plot(step_history, avg_queue_history, marker='o', label="Average Queue Length")
plt.title("Average Queue Length over Time (Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Average Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/multiintersection-qlearning-queuelength.png")
plt.show()

# Plot waiting time
plt.figure(figsize=(8, 6))
plt.plot(step_history, avg_waiting_time_history, marker='o', label="Average Waiting Time")
plt.title("Average Waiting Time over Time (Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Average Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/multiintersection-qlearning-waitingtime.png")
plt.show()
