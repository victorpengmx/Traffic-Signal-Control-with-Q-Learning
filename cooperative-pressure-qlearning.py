# cooperative q-learning with normalized pressure reward
import json
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import traci

DIRECTORY_PATH = "results"
os.makedirs(DIRECTORY_PATH, exist_ok=True)

# SUMO environment setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

SUMO_BINARY = os.environ.get("SUMO_BINARY", "sumo")
sumo_config = [
    SUMO_BINARY, '-c', 'networks/multi_cross/multi_cross.sumocfg',
    '--step-length', '0.10', '--start', '--quit-on-end'
]

traci.start(sumo_config)
if SUMO_BINARY.endswith("gui"):
    traci.gui.setSchema("View #0", "real world")

# Parameters
SIMULATION_STEPS = 10000
STEP_LENGTH = 0.1
ACTIONS = [0, 1]
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
MIN_GREEN_TIME = 20
MIN_STEPS_PER_PHASE = int(MIN_GREEN_TIME / STEP_LENGTH)
BETA = 0.5          # neighbor cooperation weight
LAMBDA_SWITCH = 0.5 # penalty for switching action
SIGMA_SMOOTH = 0.9  # smoothing factor for normalization

NEIGHBORS = {
    "Intersection1": ["Intersection2", "Intersection3"],
    "Intersection2": ["Intersection1"],
    "Intersection3": ["Intersection1"]
}

traffic_lights = traci.trafficlight.getIDList()
print(f"Detected {len(traffic_lights)} traffic lights: {traffic_lights}")

phase_step_counter = {tl: 0 for tl in traffic_lights}
Q_table = {tl: {} for tl in traffic_lights}

incoming_lanes_map = {tl: tuple(traci.trafficlight.getControlledLanes(tl)) for tl in traffic_lights}
outgoing_lanes_map = {}
for tl in traffic_lights:
    links = traci.trafficlight.getControlledLinks(tl)
    outgoing = []
    for link_group in links:
        for link in link_group:
            out_lane = link[1]
            if out_lane is not None:
                outgoing.append(out_lane)
    outgoing_lanes_map[tl] = tuple(outgoing)

lane_detectors_map = {}
for det in traci.lanearea.getIDList():
    try:
        lane = traci.lanearea.getLaneID(det)
    except traci.TraCIException:
        continue
    lane_detectors_map.setdefault(lane, []).append(det)

system_step = []
system_total_queue = []
system_total_wait = []
system_total_reward = []

def save_metric_history(tag, metric, steps, values):
    path = os.path.join(DIRECTORY_PATH, f"{tag}-{metric}.json")
    with open(path, "w") as fh:
        json.dump({"steps": steps, "values": values}, fh)

def lane_queue_for_single_lane(lane_id):
    detectors = lane_detectors_map.get(lane_id)
    if detectors:
        return sum(traci.lanearea.getLastStepVehicleNumber(det) for det in detectors)
    return traci.lane.getLastStepHaltingNumber(lane_id)

def get_lane_queue_length(lanes):
    if isinstance(lanes, (list, tuple, set)):
        return sum(lane_queue_for_single_lane(l) for l in lanes)
    return lane_queue_for_single_lane(lanes)

def get_state(tl):
    incoming_lanes = incoming_lanes_map[tl]
    detectors = []
    for lane in incoming_lanes:
        detectors.extend(lane_detectors_map.get(lane, []))

    lane_queues = []
    if detectors:
        for det in sorted(detectors):
            lane_queues.append(int(round(traci.lanearea.getLastStepVehicleNumber(det))))
    else:
        for lane in incoming_lanes:
            lane_queues.append(traci.lane.getLastStepHaltingNumber(lane))

    current_phase = traci.trafficlight.getPhase(tl)
    return tuple(lane_queues + [current_phase])

def ensure_state(tl_id, state):
    if state not in Q_table[tl_id]:
        Q_table[tl_id][state] = np.zeros(len(ACTIONS))

def choose_action(tl_id, state):
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    ensure_state(tl_id, state)
    return int(np.argmax(Q_table[tl_id][state]))

def update_q_value(tl_id, state, action, reward, next_state):
    ensure_state(tl_id, state)
    ensure_state(tl_id, next_state)
    Q_table[tl_id][state][action] += ALPHA * (
        reward + GAMMA * np.max(Q_table[tl_id][next_state]) - Q_table[tl_id][state][action]
    )

def get_total_waiting_time(tl_id):
    lanes = incoming_lanes_map[tl_id]
    vehs = [v for lane in lanes for v in traci.lane.getLastStepVehicleIDs(lane)]
    return sum(traci.vehicle.getWaitingTime(v) for v in vehs)

def lane_average_delay(lane_id):
    veh_ids = traci.lane.getLastStepVehicleIDs(lane_id)
    if not veh_ids:
        return 0.0
    waits = [traci.vehicle.getWaitingTime(veh) for veh in veh_ids]
    return float(np.mean(waits)) if waits else 0.0

def compute_movement_pressure():
    movement_pressures = {}
    per_movement_values = []

    controlled_links_cache = {
        tl: traci.trafficlight.getControlledLinks(tl)
        for tl in traffic_lights
    }

    for tl in traffic_lights:
        dmp = 0.0
        for link_group in controlled_links_cache[tl]:
            for link in link_group:
                in_lane, out_lane, _ = link
                if in_lane is None or out_lane is None:
                    continue
                flow_in = traci.lane.getLastStepVehicleNumber(in_lane)
                flow_out = traci.lane.getLastStepVehicleNumber(out_lane)
                delay = lane_average_delay(in_lane)
                movement_term = (flow_in - flow_out) * delay
                dmp += movement_term
                per_movement_values.append(abs(movement_term))
        movement_pressures[tl] = dmp

    if len(per_movement_values) > 1:
        sigma = np.std(per_movement_values)
    elif per_movement_values:
        sigma = per_movement_values[0]
    else:
        sigma = 1.0
    return movement_pressures, max(sigma, 1e-3)

sigma_dmp_smoothed = 1.0

print("\n=== Cooperative Q-Learning with Normalized Pressure Reward ===")

for step in range(SIMULATION_STEPS):
    if traci.simulation.getMinExpectedNumber() <= 0:
        break
    traci.simulationStep()

    dmp_values, sigma_dmp = compute_movement_pressure()
    sigma_dmp_smoothed = SIGMA_SMOOTH * sigma_dmp_smoothed + (1 - SIGMA_SMOOTH) * sigma_dmp
    denom = max(sigma_dmp_smoothed, 1e-3)
    normalized_dmp = {tl: dmp_values[tl] / denom for tl in traffic_lights}

    total_reward = 0
    total_queue = 0
    total_wait = 0

    for tl_id in traffic_lights:
        state = get_state(tl_id)
        action = choose_action(tl_id, state)
        switched = False

        if action == 1 and phase_step_counter[tl_id] >= MIN_STEPS_PER_PHASE:
            current_phase = traci.trafficlight.getPhase(tl_id)
            n_phases = len(traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0].phases)
            traci.trafficlight.setPhase(tl_id, (current_phase + 1) % n_phases)
            phase_step_counter[tl_id] = 0
            switched = True
        else:
            phase_step_counter[tl_id] += 1

        coop_term = normalized_dmp[tl_id]
        neighbors = NEIGHBORS.get(tl_id, [])
        if neighbors:
            coop_term += BETA * sum(normalized_dmp[nb] for nb in neighbors)

        reward = -coop_term - (LAMBDA_SWITCH if switched else 0.0)

        next_state = get_state(tl_id)
        update_q_value(tl_id, state, action, reward, next_state)

        total_reward += reward
        total_queue += get_lane_queue_length(incoming_lanes_map[tl_id])
        total_wait += get_total_waiting_time(tl_id)

    system_step.append(step)
    system_total_queue.append(total_queue)
    system_total_wait.append(total_wait)
    system_total_reward.append(total_reward)

    if step % 100 == 0:
        print(f"Step {step}: TotalQueue={total_queue}, TotalWait={total_wait:.2f}, TotalReward={total_reward:.2f}")

traci.close(False)
print("\nSimulation complete. SUMO closed automatically.")

save_metric_history("cooperative-pressure-qlearning", "queue", system_step, system_total_queue)
save_metric_history("cooperative-pressure-qlearning", "waiting", system_step, system_total_wait)

plt.figure(figsize=(8, 6))
plt.plot(system_step, system_total_queue, label="Total Queue Length")
plt.title("Queue Length (Cooperative Pressure Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DIRECTORY_PATH, "cooperative-pressure-qlearning-queuelength.png"))
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(system_step, system_total_wait, label="Total Waiting Time")
plt.title("Waiting Time (Cooperative Pressure Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Total Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DIRECTORY_PATH, "cooperative-pressure-qlearning-waitingtime.png"))
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(system_step, system_total_reward, label="Total System Reward")
plt.title("Reward (Cooperative Pressure Q-Learning)")
plt.xlabel("Simulation Step")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(DIRECTORY_PATH, "cooperative-pressure-qlearning-reward.png"))
plt.show()
