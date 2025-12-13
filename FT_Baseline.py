import os
import sys
import traci
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SUMO PATH SETUP ---
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

base_path = "Mobility Demand And SUMO/RL_Traffic_Control"
config_file = os.path.join(base_path, "RL.sumocfg") 

# --- 2. TIMING & CONSTANTS ---
MAX_SIM_TIME = 3600    # Full 1 Hour Baseline
STEP_LENGTH = 0.1      
TOTAL_STEPS = int(MAX_SIM_TIME / STEP_LENGTH) 
PHASE_DURATION_S = 30  # Standard fixed cycle
PHASE_STEPS = int(PHASE_DURATION_S / STEP_LENGTH) 

# --- 3. HELPER FUNCTIONS ---
def get_queue_length(detector_id):
    """Retrieves vehicle count from specific detector."""
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_state_detailed():
    """Groups detectors into logical approaches for analysis."""
    q_EB = sum([get_queue_length(f"Node1_2_EB_{i}_QUEUE") for i in range(3)])
    q_ST = sum([get_queue_length(f"Node2_7_SB_{i}_QUEUE") for i in range(3)])
    q_SB = sum([get_queue_length(f"Node2_5_SB_{i}_QUEUE") for i in range(3)])
    spill = sum([get_queue_length(f"NODE5_WEST_OUT_SPILL_{i}") for i in range(3)] + 
                [get_queue_length(f"NODE5_EAST_OUT_SPILL_{i}") for i in range(3)])
    return q_EB, q_ST, q_SB, spill

# --- 4. EXECUTION LOOP ---
def run_baseline():
    Sumo_config = ['sumo-gui', '-c', config_file, '--step-length', str(STEP_LENGTH), '--start']
    traci.start(Sumo_config)

    hist = {
        "time": [], "q_EB": [], "q_ST": [], "q_SB": [], 
        "spill": [], "reward": [], "co2": [], "wait": [], "arrived": []
    }
    
    cumulative_reward = 0.0
    total_arrived = 0

    print(f"\n=== Running FT Baseline ({MAX_SIM_TIME}s) ===")

    for step in range(TOTAL_STEPS):
        # Fixed-Time Cycle Logic
        if step % PHASE_STEPS == 0:
            current_p = traci.trafficlight.getPhase("Node2")
            traci.trafficlight.setPhase("Node2", (current_p + 1) % 4)

        traci.simulationStep()
        
        # Log data every 1 second
        if step % 10 == 0:
            q_eb, q_st, q_sb, spl = get_state_detailed()
            
            # Metric Calculation
            rew = -(q_eb + q_st + q_sb + spl)
            cumulative_reward += rew
            total_arrived += traci.simulation.getArrivedNumber()
            
            co2 = sum([traci.lane.getCO2Emission(l) for l in traci.lane.getIDList()]) / 1000000.0
            wait = sum([traci.lane.getWaitingTime(l) for l in traci.lane.getIDList()])

            hist["time"].append(step * STEP_LENGTH)
            hist["q_EB"].append(q_eb)
            hist["q_ST"].append(q_st)
            hist["q_SB"].append(q_sb)
            hist["spill"].append(spl)
            hist["reward"].append(cumulative_reward)
            hist["co2"].append(co2)
            hist["wait"].append(wait)

            if step % 6000 == 0:
                print(f"Progress: {int(step/TOTAL_STEPS*100)}% | Vehicles Arrived: {total_arrived}")

    traci.close()
    
    # --- FINAL COMPREHENSIVE SUMMARY ---
    print("\n" + "="*50)
    print("         FIXED-TIME PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Total Arrived Vehicles:   {total_arrived}")
    print(f"Max Eastbound Queue:      {max(hist['q_EB'])} veh")
    print(f"Max SB Top Queue:         {max(hist['q_ST'])} veh")
    print(f"Max SB Bottom Queue:      {max(hist['q_SB'])} veh")
    print(f"Peak Bridge Spillback:    {max(hist['spill'])} veh")
    print(f"Total Network Delay:      {sum(hist['wait']):.2f} veh-s")
    print(f"Total CO2 Emissions:      {sum(hist['co2']):.2f} kg")
    print(f"Final System Score:       {cumulative_reward:.2f}")
    print("="*50)

    return hist

# --- 5. ENHANCED PLOTTING ---
def plot_results(h):
    def smooth(data, window=25):
        return np.convolve(data, np.ones(window)/window, mode='same')

    # Figure 1: Load Analysis
    plt.figure("FT Traffic Load", figsize=(12, 6))
    plt.plot(h["time"], smooth(h["q_EB"]), label='Eastbound (Node 1->2)', linewidth=2)
    plt.plot(h["time"], smooth(h["q_ST"]), label='SB Top (Node 7->2)', linewidth=2)
    plt.plot(h["time"], smooth(h["q_SB"]), label='SB Bot (Node 2->5)', linewidth=2)
    plt.plot(h["time"], smooth(h["spill"]), label='Exit Spillback (Node 5)', color='black', linewidth=3, linestyle='--')
    plt.title('Baseline Congestion (1-Hour Smoothed)', fontsize=14)
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Vehicles')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figure 2: Delay & Environment
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax1.plot(h["time"], smooth(h["wait"]), color='orange', linewidth=2)
    ax1.set_title('Total Waiting Time (Fixed-Time)')
    ax1.set_ylabel('Seconds')
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(h["time"], smooth(h["co2"]), color='green', alpha=0.15)
    ax2.plot(h["time"], smooth(h["co2"]), color='green', linewidth=2)
    ax2.set_title('CO2 Emission Rate')
    ax2.set_ylabel('kg/s')
    ax2.set_xlabel('Time (Seconds)')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figure 3: System Health
    plt.figure("FT Performance Score", figsize=(12, 4))
    plt.plot(h["time"], h["reward"], color='purple', linewidth=2)
    plt.title('Baseline Cumulative Reward (System Health)')
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    results = run_baseline()
    plot_results(results)
