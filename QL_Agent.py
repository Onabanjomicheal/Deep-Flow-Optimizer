import os
import sys
import random
import traci
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SUMO SETUP ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

base_path = "Mobility Demand And SUMO/RL_Traffic_Control"
config_file = os.path.join(base_path, "RL.sumocfg") 

# --- 2. HYPERPARAMETERS & CONSTANTS ---
MAX_SIM_TIME = 3600    
STEP_LENGTH = 0.1
TOTAL_STEPS = int(MAX_SIM_TIME / STEP_LENGTH)

ALPHA = 0.1 
GAMMA = 0.9 
EPSILON = 0.1
ACTIONS = [0, 1] 
Q_table = {}

MIN_GREEN_STEPS = 100 
last_switch_step = -MIN_GREEN_STEPS

# --- 3. HELPER FUNCTIONS ---
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_state_detailed():
    q_eb = sum([get_queue_length(f"Node1_2_EB_{i}_QUEUE") for i in range(3)])
    q_st = sum([get_queue_length(f"Node2_7_SB_{i}_QUEUE") for i in range(3)])
    q_sb = sum([get_queue_length(f"Node2_5_SB_{i}_QUEUE") for i in range(3)])
    spill = sum([get_queue_length(f"NODE5_WEST_OUT_SPILL_{i}") for i in range(3)] + 
                [get_queue_length(f"NODE5_EAST_OUT_SPILL_{i}") for i in range(3)])
    return q_eb, q_st, q_sb, spill

def get_reward(state_tuple):
    return -float(sum(state_tuple[:-1]))

def get_action_from_policy(state):
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))
    return int(np.argmax(Q_table[state]))

def apply_action(action, step):
    global last_switch_step
    if action == 1 and (step - last_switch_step >= MIN_GREEN_STEPS):
        current_p = traci.trafficlight.getPhase("Node2")
        next_p = (current_p + 1) % 4 
        traci.trafficlight.setPhase("Node2", next_p)
        last_switch_step = step

def update_Q_table(old_state, action, reward, new_state):
    if old_state not in Q_table: Q_table[old_state] = np.zeros(len(ACTIONS))
    if new_state not in Q_table: Q_table[new_state] = np.zeros(len(ACTIONS))
    old_q = Q_table[old_state][action]
    best_future_q = np.max(Q_table[new_state])
    Q_table[old_state][action] = old_q + ALPHA * (reward + GAMMA * best_future_q - old_q)

# --- 4. MAIN TRAINING LOOP ---
def run_ql_simulation():
    Sumo_config = ['sumo-gui', '-c', config_file, '--step-length', str(STEP_LENGTH), '--start']
    traci.start(Sumo_config)

    hist = {
        "time": [], "q_EB": [], "q_ST": [], "q_SB": [], 
        "spill": [], "reward": [], "co2": [], "wait": []
    }
    cumulative_reward = 0.0
    total_arrived = 0

    print("\n=== Starting Q-Learning Agent Training (3600s) ===")
    
    for step in range(TOTAL_STEPS):
        # Current detailed state
        q_eb, q_st, q_sb, spill = get_state_detailed()
        current_phase = traci.trafficlight.getPhase("Node2")
        state = (q_eb, q_st, q_sb, spill, current_phase)

        action = get_action_from_policy(state)
        apply_action(action, step)
        
        traci.simulationStep()
        
        # New state and learning
        new_q_eb, new_q_st, new_q_sb, new_spill = get_state_detailed()
        new_phase = traci.trafficlight.getPhase("Node2")
        new_state = (new_q_eb, new_q_st, new_q_sb, new_spill, new_phase)
        
        reward = get_reward(new_state)
        cumulative_reward += reward
        update_Q_table(state, action, reward, new_state)

        # Log Metrics every 1 second
        if step % 10 == 0:
            total_arrived += traci.simulation.getArrivedNumber()
            co2 = sum([traci.lane.getCO2Emission(l) for l in traci.lane.getIDList()]) / 1000000.0
            wait = sum([traci.lane.getWaitingTime(l) for l in traci.lane.getIDList()])

            hist["time"].append(step * STEP_LENGTH)
            hist["q_EB"].append(q_eb)
            hist["q_ST"].append(q_st)
            hist["q_SB"].append(q_sb)
            hist["spill"].append(spill)
            hist["reward"].append(cumulative_reward)
            hist["co2"].append(co2)
            hist["wait"].append(wait)

        if step % 6000 == 0:
            print(f"Time: {int(step*0.1)}s | Progress: {int(step/TOTAL_STEPS*100)}% | Q-Table: {len(Q_table)}")

    traci.close()

    # --- FINAL SUMMARY (MATCHES BASELINE FORMAT) ---
    print("\n" + "="*50)
    print("         FINAL Q-LEARNING PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Vehicles Arrived:   {total_arrived}")
    print(f"Max Queue (EB):           {max(hist['q_EB'])} vehicles")
    print(f"Max Queue (SB Top):       {max(hist['q_ST'])} vehicles")
    print(f"Max Queue (SB Bot):       {max(hist['q_SB'])} vehicles")
    print(f"Peak Spillback:           {max(hist['spill'])} vehicles")
    print(f"Total Network Delay:      {sum(hist['wait']):.2f} vehicle-seconds")
    print(f"Total CO2 Emissions:      {sum(hist['co2']):.2f} kg")
    print(f"Final Performance Score:  {cumulative_reward:.2f}")
    print(f"Unique States Discovered: {len(Q_table)}")
    print("="*50)

    return hist, total_arrived

# --- 5. CLEAN PLOTTING ---
def plot_ql_results(h, arrived):
    def smooth(data, weight=0.95):
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    plt.figure("QL Detailed Load", figsize=(12, 6))
    plt.plot(h["time"], smooth(h["q_EB"]), label='Eastbound')
    plt.plot(h["time"], smooth(h["q_ST"]), label='SB Top')
    plt.plot(h["time"], smooth(h["q_SB"]), label='SB Bot')
    plt.plot(h["time"], smooth(h["spill"]), label='Spillback', color='black', linewidth=2, linestyle='--')
    plt.title(f'Q-Learning Congestion Trend (Arrived: {arrived})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(h["time"], smooth(h["wait"]), color='orange', linewidth=2)
    ax1.set_title('QL Total Network Delay')
    ax1.set_ylabel('Wait Time (s)')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(h["time"], smooth(h["co2"]), color='green', alpha=0.2)
    ax2.plot(h["time"], smooth(h["co2"]), color='green', linewidth=2)
    ax2.set_title('QL CO2 Emission Rate')
    ax2.set_ylabel('kg/s')
    ax2.set_xlabel('Time (Seconds)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.figure("QL Reward Performance", figsize=(12, 4))
    plt.plot(h["time"], h["reward"], color='purple', linewidth=2)
    plt.title('QL Cumulative Reward')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    history, total_arrived = run_ql_simulation()
    plot_ql_results(history, total_arrived)