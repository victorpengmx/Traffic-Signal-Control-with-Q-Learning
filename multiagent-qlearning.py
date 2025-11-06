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

INTERSECTIONS = [
    "Intersection1", "Intersection2", "Intersection3"
]

# Neighbors for cooperative info sharing
NEIGHBORS = {
    "Intersection1": ["Intersection2", "Intersection3"],
    "Intersection2": ["Intersection1"],
    "Intersection3": ["Intersection1"]
}

traffic_lights = traci.trafficlight.getIDList()
print(f"Detected {len(traffic_lights)} traffic lights: {traffic_lights}")

phase_step_counter = {tl: 0 for tl in traffic_lights}

# Initialize Q-tables
Q = {tl: {} for tl in INTERSECTIONS}

# data
system_step = []
system_total_queue = []
system_total_wait = []
system_total_reward = []

def get_lane_queue_length(lane_ids):
    # Return total number of halted vehicles for given lanes
    return sum(traci.lane.getLastStepHaltingNumber(l) for l in lane_ids)

def get_state(tl_id):
    # Define state as discretized total queue length (rounded to nearest 5 vehicles)
    incoming_lanes = traci.trafficlight.getControlledLanes(tl_id)
    queue = get_lane_queue_length(incoming_lanes)
    return int(round(queue / 5.0) * 5)

def choose_action(tl_id, state):
    # Epsilon-greedy policy
    if random.uniform(0, 1) < EPSILON or state not in Q[tl_id]:
        return random.choice(ACTIONS)
    return np.argmax(Q[tl_id][state])

def update_q_value(tl_id, state, action, reward, next_state):
    if state not in Q[tl_id]:
        Q[tl_id][state] = np.zeros(len(ACTIONS))
    if next_state not in Q[tl_id]:
        Q[tl_id][next_state] = np.zeros(len(ACTIONS))
    Q[tl_id][state][action] += ALPHA * (
        reward + GAMMA * np.max(Q[tl_id][next_state]) - Q[tl_id][state][action]
    )

def get_total_waiting_time(tl_id):
    # Return total waiting time for vehicles approaching traffic light
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    vehs = [v for l in lanes for v in traci.lane.getLastStepVehicleIDs(l)]
    return sum(traci.vehicle.getWaitingTime(v) for v in vehs)

def cooperative_reward(tl_id):
    # Reward = negative of (own waiting + neighbor queue)
    local_wait = get_total_waiting_time(tl_id)
    neighbor_queues = []
    for nb in NEIGHBORS.get(tl_id, []):
        n_lanes = traci.trafficlight.getControlledLanes(nb)
        n_queue = get_lane_queue_length(n_lanes)
        neighbor_queues.append(n_queue)
    avg_neighbor_queue = np.mean(neighbor_queues) if neighbor_queues else 0
    return - (local_wait + 0.5 * avg_neighbor_queue)

print("\n=== Cooperative Multi-Agent Q-Learning Traffic Control ===")

# Simulation loop
for step in range(SIMULATION_STEPS):
    if traci.simulation.getMinExpectedNumber() <= 0:
        break

    traci.simulationStep()

    total_reward = 0
    total_queue = 0
    total_wait = 0

    for tl_id in INTERSECTIONS:
        state = get_state(tl_id)
        action = choose_action(tl_id, state)

        # Execute action
        if action == 1 and phase_step_counter[tl_id] >= MIN_STEPS_PER_PHASE:
            current_phase = traci.trafficlight.getPhase(tl_id)
            n_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases)
            traci.trafficlight.setPhase(tl_id, (current_phase + 1) % n_phases)

        # Compute cooperative reward
        reward = cooperative_reward(tl_id)
        next_state = get_state(tl_id)
        update_q_value(tl_id, state, action, reward, next_state)

        # Aggregate metrics
        total_reward += reward
        total_queue += get_lane_queue_length(traci.trafficlight.getControlledLanes(tl_id))
        total_wait += get_total_waiting_time(tl_id)

    # Record data
    system_step.append(step)
    system_total_queue.append(total_queue)
    system_total_wait.append(total_wait)
    system_total_reward.append(total_reward)

    if step % 100 == 0:
        print(f"Step {step}: TotalQueue={total_queue}, TotalWait={total_wait:.2f}, TotalReward={total_reward:.2f}")

# shutdown SUMO
traci.close(False)
print("\nSimulation complete. SUMO closed automatically.")

# Plots
plt.figure(figsize=(10, 6))
plt.plot(system_step, system_total_queue, label="Total Queue Length")
plt.title("System-Wide Queue Length (Cooperative Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "cooperative-qlearning-queuelength.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(system_step, system_total_wait, label="Total Waiting Time")
plt.title("System-Wide Waiting Time (Cooperative Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "cooperative-qlearning-waitingtime.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(system_step, system_total_reward, label="Total System Reward")
plt.title("System-Wide Reward (Cooperative Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "cooperative-qlearning-reward.png")
plt.show()
