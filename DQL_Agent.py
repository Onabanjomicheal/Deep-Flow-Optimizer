import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
import traci

# --- 1. SUMO SETUP ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

base_path = "Mobility Demand And SUMO/RL_Traffic_Control"
config_file = os.path.join(base_path, "RL.sumocfg") 

# --- 2. HYPERPARAMETERS ---
MAX_SIM_TIME = 3600    
STEP_LENGTH = 0.1
TOTAL_STEPS = int(MAX_SIM_TIME / STEP_LENGTH)
DECISION_INTERVAL = 50 

ALPHA = 0.001          
GAMMA = 0.95           
EPSILON = 0.1          
ACTIONS = [0, 1]       

# --- 3. DQL MODEL ---
def build_model(state_size, action_size):
    model = keras.Sequential([
        layers.Input(shape=(state_size,)),
        layers.Dense(48, activation='relu'),
        layers.Dense(48, activation='relu'),
        layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=ALPHA))
    return model

state_size = 12
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)

# --- 4. HELPER FUNCTIONS ---
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_state():
    q_eb = [get_queue_length(f"Node1_2_EB_{i}_QUEUE") for i in range(3)]
    q_st = [get_queue_length(f"Node2_7_SB_{i}_QUEUE") for i in range(3)]
    q_sm = [get_queue_length(f"Node2_5_SB_{i}_QUEUE") for i in range(3)]
    spill_W = sum([get_queue_length(f"NODE5_WEST_OUT_SPILL_{i}") for i in range(3)])
    spill_E = sum([get_queue_length(f"NODE5_EAST_OUT_SPILL_{i}") for i in range(3)])
    current_phase = traci.trafficlight.getPhase("Node2")
    state = q_eb + q_st + q_sm + [spill_W, spill_E, current_phase]
    return np.array(state, dtype=np.float32).reshape((1, -1))

def get_reward(state_vec):
    return -float(np.sum(state_vec[0][:-1]))

# --- 5. EXECUTION LOOP ---
def run_dql_simulation():
    Sumo_config = ['sumo-gui', '-c', config_file, '--step-length', str(STEP_LENGTH), '--start']
    traci.start(Sumo_config)

    hist = {
        "time": [], "q_EB": [], "q_ST": [], "q_SB": [], 
        "spill": [], "reward": [], "co2": [], "wait": []
    }
    cumulative_reward = 0.0
    total_arrived = 0
    last_state = None
    last_action = None

    print("\n=== Starting Optimized Deep Q-Learning (3600s) ===")

    for step in range(TOTAL_STEPS):
        if step % DECISION_INTERVAL == 0:
            current_state = get_state()
            if last_state is not None:
                reward = get_reward(current_state)
                cumulative_reward += reward
                target_q = dqn_model.predict(last_state, verbose=0)
                next_q = dqn_model.predict(current_state, verbose=0)
                target_q[0][last_action] = reward + GAMMA * np.max(next_q[0])
                dqn_model.fit(last_state, target_q, epochs=1, verbose=0)

            if random.random() < EPSILON:
                action = random.choice(ACTIONS)
            else:
                action = np.argmax(dqn_model.predict(current_state, verbose=0)[0])
            
            if action == 1:
                curr = traci.trafficlight.getPhase("Node2")
                traci.trafficlight.setPhase("Node2", (curr + 1) % 4)
            
            last_state = current_state
            last_action = action

        traci.simulationStep()
        
        if step % 10 == 0:
            total_arrived += traci.simulation.getArrivedNumber()
            co2 = sum([traci.lane.getCO2Emission(l) for l in traci.lane.getIDList()]) / 1000000.0
            wait = sum([traci.lane.getWaitingTime(l) for l in traci.lane.getIDList()])
            
            # Use current state for history logging
            current_s = get_state()[0]
            hist["time"].append(step * STEP_LENGTH)
            hist["q_EB"].append(sum(current_s[0:3]))
            hist["q_ST"].append(sum(current_s[3:6]))
            hist["q_SB"].append(sum(current_s[6:9]))
            hist["spill"].append(current_s[9] + current_s[10])
            hist["reward"].append(cumulative_reward)
            hist["co2"].append(co2)
            hist["wait"].append(wait)

        if step % 6000 == 0:
            print(f"Time: {int(step*0.1)}s | Arrived: {total_arrived} | Speed: Optimized")

    traci.close()
    
    # --- FINAL SUMMARY (MATCHES FT & QL FORMAT) ---
    print("\n" + "="*50)
    print("         FINAL DQL PERFORMANCE REPORT")
    print("="*50)
    print(f"Total Simulation Time:    {MAX_SIM_TIME} seconds")
    print(f"Total Vehicles Arrived:   {total_arrived}")
    print(f"Max Queue (EB):           {max(hist['q_EB'])} vehicles")
    print(f"Max Queue (SB Top):       {max(hist['q_ST'])} vehicles")
    print(f"Max Queue (SB Bot):       {max(hist['q_SB'])} vehicles")
    print(f"Peak Spillback:           {max(hist['spill'])} vehicles")
    print(f"Total Network Delay:      {sum(hist['wait']):.2f} vehicle-seconds")
    print(f"Total CO2 Emissions:      {sum(hist['co2']):.2f} kg")
    print(f"Final Performance Score:  {cumulative_reward:.2f}")
    print("="*50)

    return hist, total_arrived

# --- 6. THREE PROFESSIONAL GRAPHS ---
def plot_dql_results(h, arrived):
    def smooth(data, weight=0.95):
        last = data[0]
        smoothed = []
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    # Graph 1: Detailed Load Analysis
    plt.figure("DQL Traffic Load", figsize=(12, 6))
    plt.plot(h["time"], smooth(h["q_EB"]), label='Eastbound')
    plt.plot(h["time"], smooth(h["q_ST"]), label='SB Top')
    plt.plot(h["time"], smooth(h["q_SB"]), label='SB Bot')
    plt.plot(h["time"], smooth(h["spill"]), label='Spillback', color='black', linewidth=2, linestyle='--')
    plt.title(f'DQL Congestion Trends (Arrived: {arrived})')
    plt.xlabel('Time (s)'); plt.ylabel('Vehicles')
    plt.legend(loc='upper left'); plt.grid(True, alpha=0.3); plt.tight_layout()

    # Graph 2: Stacked Delay and CO2
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(h["time"], smooth(h["wait"]), color='orange', linewidth=2)
    ax1.set_title('DQL Network-Wide Delay')
    ax1.set_ylabel('Wait Time (s)'); ax1.grid(True, alpha=0.3)

    ax2.fill_between(h["time"], smooth(h["co2"]), color='green', alpha=0.2)
    ax2.plot(h["time"], smooth(h["co2"]), color='green', linewidth=2)
    ax2.set_title('DQL CO2 Emission Rate')
    ax2.set_ylabel('kg/s'); ax2.set_xlabel('Time (s)'); ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Graph 3: Cumulative Reward (System Health)
    plt.figure("DQL System Score", figsize=(12, 4))
    plt.plot(h["time"], h["reward"], color='purple', linewidth=2)
    plt.title('DQL Cumulative Reward (Learning Progress)')
    plt.xlabel('Time (s)'); plt.ylabel('Score'); plt.grid(True, alpha=0.3); plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    history, total_arrived = run_dql_simulation()
    plot_dql_results(history, total_arrived)