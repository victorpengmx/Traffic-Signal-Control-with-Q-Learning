# cooperative q-learning with shared reward
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
    'sumo-gui', '-c', 'networks/multi_cross/multi_cross.sumocfg',
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

MIN_GREEN_TIME = 20
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
Q_table = {tl: {} for tl in traffic_lights}

# data
system_step = []
system_total_queue = []
system_total_wait = []
system_total_reward = []

# def get_lane_queue_length(lane_id):
#     return traci.lanearea.getLastStepVehicleNumber(lane_id)

def get_lane_queue_length(lanes):
    # lanes may be a list, tuple, or string
    if isinstance(lanes, (list, tuple, set)):
        return sum(get_lane_queue_length(l) for l in lanes)
    # single lane string
    detectors = [d for d in traci.lanearea.getIDList()
                 if traci.lanearea.getLaneID(d) == lanes]
    if len(detectors) == 0:
        # fallback: halting vehicles
        return traci.lane.getLastStepHaltingNumber(lanes)
    return sum(traci.lanearea.getLastStepVehicleNumber(d) for d in detectors)

def compute_pressure(tl):
    # Incoming lanes
    incoming = traci.trafficlight.getControlledLanes(tl)
    Qin = get_lane_queue_length(incoming)

    # Outgoing lanes from link index lists
    links = traci.trafficlight.getControlledLinks(tl)
    outgoing = []
    for link_group in links:
        for link in link_group:
            out_lane = link[1]  # (inLane, outLane, via)
            if out_lane is not None:
                outgoing.append(out_lane)

    Qout = get_lane_queue_length(outgoing)

    return Qin - Qout

def get_state(tl):
    incoming_lanes = set(traci.trafficlight.getControlledLanes(tl))
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

    current_phase = traci.trafficlight.getPhase(tl)
    state = tuple(lane_queues + [current_phase])
    return state

def choose_action(tl_id, state):
    # Epsilon-greedy policy
    if random.uniform(0, 1) < EPSILON or state not in Q_table[tl_id]:
        return random.choice(ACTIONS)
    return np.argmax(Q_table[tl_id][state])

def update_q_value(tl_id, state, action, reward, next_state):
    if state not in Q_table[tl_id]:
        Q_table[tl_id][state] = np.zeros(len(ACTIONS))
    if next_state not in Q_table[tl_id]:
        Q_table[tl_id][next_state] = np.zeros(len(ACTIONS))
    Q_table[tl_id][state][action] += ALPHA * (
        reward + GAMMA * np.max(Q_table[tl_id][next_state]) - Q_table[tl_id][state][action]
    )

def get_total_waiting_time(tl_id):
    # Return total waiting time for vehicles approaching traffic light
    lanes = traci.trafficlight.getControlledLanes(tl_id)
    vehs = [v for l in lanes for v in traci.lane.getLastStepVehicleIDs(l)]
    return sum(traci.vehicle.getWaitingTime(v) for v in vehs)

def cooperative_reward(tl_id):
    beta = 0.5  # cooperation weight

    # Local pressure
    P_local = compute_pressure(tl_id)

    # Neighbor pressures
    neighbor_pressures = []
    for nb in NEIGHBORS.get(tl_id, []):
        neighbor_pressures.append(compute_pressure(nb))

    P_neighbors = np.mean(neighbor_pressures) if neighbor_pressures else 0

    # Cooperative pressure reward
    return - (P_local + beta * P_neighbors)


print("\n=== Cooperative Multi-Agent Q-Learning Traffic Control ===")

# Simulation loop
for step in range(SIMULATION_STEPS):
    if traci.simulation.getMinExpectedNumber() <= 0:
        break
    traci.simulationStep()

    total_reward = 0
    total_queue = 0
    total_wait = 0

    # for tl_id in INTERSECTIONS:
    for tl_id in traffic_lights:
        state = get_state(tl_id)
        action = choose_action(tl_id, state)

        # Execute action
        if action == 1 and phase_step_counter[tl_id] >= MIN_STEPS_PER_PHASE:
            current_phase = traci.trafficlight.getPhase(tl_id)
            n_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases)
            traci.trafficlight.setPhase(tl_id, (current_phase + 1) % n_phases)
            phase_step_counter[tl_id] = 0
        else:
            phase_step_counter[tl_id] += 1

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

traci.close(False)
print("\nSimulation complete. SUMO closed automatically.")

# Plots
plt.figure(figsize=(8, 6))
plt.plot(system_step, system_total_queue, label="Total Queue Length")
plt.title("System-Wide Queue Length (Cooperative Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/cooperative-qlearning-queuelength.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(system_step, system_total_wait, label="Total Waiting Time")
plt.title("System-Wide Waiting Time (Cooperative Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/cooperative-qlearning-waitingtime.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(system_step, system_total_reward, label="Total System Reward")
plt.title("System-Wide Reward (Cooperative Multi-Agent Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/cooperative-qlearning-reward.png")
plt.show()
