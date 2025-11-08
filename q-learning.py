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

# SUMO configuration (GUI enabled, runs automatically)
sumo_config = [
    'sumo-gui', '-c', 'networks/single_cross/simple_cross_lad.sumocfg',
    '--step-length', '0.10', '--start', '--quit-on-end'
]

# Start SUMO and Traci automatically
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Parameters
SIMULATION_STEPS = 10000
TRAFFIC_LIGHT = "cluster_J1_J2_J4_J6"
STEP_LENGTH = 0.1
ACTIONS = [0, 1]  
ALPHA = 0.1      
GAMMA = 0.9       
EPSILON = 0.1     

MIN_GREEN_TIME = 20
MIN_STEPS_PER_PHASE = int(MIN_GREEN_TIME / STEP_LENGTH)

# Data
step_history = []
queue_history = []
waiting_time_history = []

# Q-table (state = simplified traffic condition, action = phase change or not)
Q_table = {}

def get_lane_queue_length(lane_id):
    return traci.lanearea.getLastStepVehicleNumber(lane_id)

def get_total_queue_length():
    total = 0
    incoming_lanes = set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT))

    detectors = [
        det for det in traci.lanearea.getIDList()
        if traci.lanearea.getLaneID(det) in incoming_lanes
    ]
    detectors.sort()

    # Collect queue length from detectors
    for det in detectors:
        q = traci.lanearea.getLastStepVehicleNumber(det)
        q_discrete = int(round(q / 1.0))
        total += q_discrete
    return total

def get_total_waiting_time():
    # Total waiting time of all vehicles
    vehicle_ids = traci.vehicle.getIDList()
    total = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    return total

def get_reward(state):
    total_queue = sum(state[:-1])  # exclude current_phase
    return -float(total_queue)

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

def choose_action(state):
    # Epsilon-greedy policy
    if random.uniform(0, 1) < EPSILON or state not in Q_table:
        return random.choice(ACTIONS)
    else:
        return np.argmax(Q_table[state])

def update_q_value(state, action, reward, next_state):
    # Initialize Q-values if not present
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))
    if next_state not in Q_table:
        Q_table[next_state] = np.zeros(len(ACTIONS))

    # Q-learning update rule
    Q_table[state][action] = Q_table[state][action] + ALPHA * (
        reward + GAMMA * np.max(Q_table[next_state]) - Q_table[state][action]
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
    action = choose_action(state)

    # Apply action (0 = keep phase, 1 = switch phase)
    if action == 1 and phase_step_counter >= MIN_STEPS_PER_PHASE:
        current_phase = (current_phase + 1) % len(
            traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT)[0].phases
        )
        traci.trafficlight.setPhase(TRAFFIC_LIGHT, current_phase)
        phase_step_counter = 0
    else:
        phase_step_counter += 1

    # Next state and Q-value update
    next_state = get_state()
    reward = get_reward(next_state)
    update_q_value(state, action, reward, next_state)

    # Record metrics every 100 steps
    if step % 100 == 0:
        total_q = get_total_queue_length()
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
plt.savefig(DIRECTORY_PATH + "/qlearning-queuelength.png")
plt.show()

# Plot Waiting Time
plt.figure(figsize=(8, 6))
plt.plot(step_history, waiting_time_history, marker='o', label="Total Waiting Time")
plt.title("Total Waiting Time over Steps (Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/qlearning-waitingtime.png")
plt.show()
