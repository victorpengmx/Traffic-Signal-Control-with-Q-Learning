import json
import os
import sys
import traci
import matplotlib.pyplot as plt

DIRECTORY_PATH = "results"
os.makedirs(DIRECTORY_PATH, exist_ok=True)

os.environ.setdefault("SUMO_HOME", "/opt/homebrew/Cellar/sumo/1.20.0/share/sumo")

# SUMO environment setup
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# SUMO configuration
sumo_config = [
    'sumo-gui', '-c', 'networks/simple_cross.sumocfg',
    '--step-length', '0.10', '--start', '--quit-on-end'
]

# Start SUMO and Traci
traci.start(sumo_config)
traci.gui.setSchema("View #0", "real world")

# Parameters
SIMULATION_STEPS = 10000
TRAFFIC_LIGHT = "cluster_J1_J2_J4_J6"
# GREEN_DURATION = 60
# STEP_LENGTH = 0.1
# STEPS_PER_PHASE = int(GREEN_DURATION / STEP_LENGTH)

# Data
step_history = []
queue_history = []
waiting_time_history = []

def save_metric_history(tag, metric, steps, values):
    """Persist history data so comparison scripts can consume it."""
    path = os.path.join(DIRECTORY_PATH, f"{tag}-{metric}.json")
    with open(path, "w") as fh:
        json.dump({"steps": steps, "values": values}, fh)

def get_total_queue_length():
    # Sum halted vehicles on each lane controlled by this traffic light
    incoming_lanes = set(traci.trafficlight.getControlledLanes(TRAFFIC_LIGHT))
    return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lanes)

def get_total_waiting_time():
    #Total waiting time of all vehicles
    vehicle_ids = traci.vehicle.getIDList()
    total = sum(traci.vehicle.getWaitingTime(vid) for vid in vehicle_ids)
    return total

# Simulation Loop
print("\n=== Running Fixed-Timing Control ===")
phase_step_counter = 0

# Traffic light phase change follows tlLogic in the .net.xml file
for step in range(SIMULATION_STEPS):
    traci.simulationStep()

    # Record metrics every 100 steps
    if step % 100 == 0:
        total_q = get_total_queue_length()
        total_w = get_total_waiting_time()
        print(f"Step {step}: Queue={total_q}, WaitingTime={total_w:.2f}")

        step_history.append(step)
        queue_history.append(total_q)
        waiting_time_history.append(total_w)

traci.close()
print("\nSimulation complete.")

save_metric_history("fixed-timing", "waiting", step_history, waiting_time_history)
save_metric_history("fixed-timing", "queue", step_history, queue_history)

plt.figure(figsize=(8, 6))
plt.plot(step_history, queue_history, marker='o', label="Total Queue Length")
plt.title("Queue Length over Steps")
plt.xlabel("Simulation Step")
plt.ylabel("Queue Length (vehicles)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/fixedtiming-queuelength.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(step_history, waiting_time_history, marker='o', label="Total Waiting Time")
plt.title("Total Waiting Time over Steps")
plt.xlabel("Simulation Step")
plt.ylabel("Waiting Time (s)")
plt.legend()
plt.grid(True)
plt.savefig(DIRECTORY_PATH + "/fixedtiming-waitingtime.png")
plt.show()
