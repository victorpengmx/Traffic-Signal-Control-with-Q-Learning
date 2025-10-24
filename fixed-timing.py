import os
import sys
import traci
import matplotlib.pyplot as plt

# SUMO environment setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
sumo_config = [
    'sumo-gui', '-c', 'networks/simple_cross.sumocfg', '--step-length', '0.10', '--delay', '1000'
]

# Start SUMO and Traci
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Parameters
SIMULATION_STEPS = 10000
TRAFFIC_LIGHT = "cluster_J1_J2_J4_J6"
GREEN_DURATION = 30
STEP_LENGTH = 0.1
STEPS_PER_PHASE = int(GREEN_DURATION / STEP_LENGTH)

# Data
step_history = []
queue_history = []
waiting_time_history = []

def get_queue_length():
    # Total number of halted vehicles (queue length)
    total = 0
    for lane_id in traci.lane.getIDList():
        total += traci.lane.getLastStepHaltingNumber(lane_id)
    return total

def get_total_waiting_time():
    #Total waiting time of all vehicles
    vehicle_ids = traci.vehicle.getIDList()
    total = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    return total

# Simulation Loop
print("\n=== Running Fixed-Timing Control ===")
phase_step_counter = 0

for step in range(SIMULATION_STEPS):
    traci.simulationStep()

    # Change phase when fixed duration elapsed
    if phase_step_counter >= STEPS_PER_PHASE:
        current_phase = traci.trafficlight.getPhase(TRAFFIC_LIGHT)
        next_phase = (current_phase + 1) % traci.trafficlight.getCompleteRedYellowGreenDefinition(TRAFFIC_LIGHT)[0].phases.__len__()
        traci.trafficlight.setPhase(TRAFFIC_LIGHT, next_phase)
        phase_step_counter = 0
    else:
        phase_step_counter += 1

    # Record metrics every 100 steps
    if step % 100 == 0:
        total_q = get_queue_length()
        total_w = get_total_waiting_time()
        print(f"Step {step}: Queue={total_q}, WaitingTime={total_w:.2f}")

        step_history.append(step)
        queue_history.append(total_q)
        waiting_time_history.append(total_w)

traci.close()
print("\nSimulation complete.")

plt.figure(figsize=(8, 6))
plt.plot(step_history, queue_history, marker='o', label="Total Queue Length")
plt.title("Queue Length over Steps")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig("queue-length.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(step_history, waiting_time_history, marker='o', label="Total Waiting Time")
plt.title("Total Waiting Time over Steps")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig("waiting-time.png")
plt.show()
