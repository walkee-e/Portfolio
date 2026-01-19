import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Function to smooth data using a moving average
def smooth_data(data, window_size=5):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Read and parse the data
def parse_data(file_path):
    total_numsteps = []
    rewards = []
    with open(file_path, 'r') as file:
        for line in file:
            if "total numsteps:" in line and "reward:" in line:
                parts = line.split(",")
                numsteps = int(parts[1].split(":")[1].strip())
                reward = float(parts[3].split(":")[1].strip())
                total_numsteps.append(numsteps)
                rewards.append(reward)
    return np.array(total_numsteps), np.array(rewards)

# Main function to load data, smooth it, and plot
def main(file_path1, file_path2, file_path3, file_path4, file_path5, file_path6):
    # Load data from the first text file
    total_numsteps1, rewards1 = parse_data(file_path1)
    smoothed_rewards1 = smooth_data(rewards1)
    total_numsteps_smoothed1 = total_numsteps1[:len(smoothed_rewards1)]

    # Load data from the second text file
    total_numsteps2, rewards2 = parse_data(file_path2)
    smoothed_rewards2 = smooth_data(rewards2)
    total_numsteps_smoothed2 = total_numsteps2[:len(smoothed_rewards2)]
    
    # Load data from the third text file
    total_numsteps3, rewards3 = parse_data(file_path3)
    smoothed_rewards3 = smooth_data(rewards3)
    total_numsteps_smoothed3 = total_numsteps3[:len(smoothed_rewards3)]
    
    # Load data from the fourth text file
    total_numsteps4, rewards4 = parse_data(file_path4)
    smoothed_rewards4 = smooth_data(rewards4)
    total_numsteps_smoothed4 = total_numsteps4[:len(smoothed_rewards4)]
    
    # Load data from the fifth text file
    total_numsteps5, rewards5 = parse_data(file_path5)
    smoothed_rewards5 = smooth_data(rewards5)
    total_numsteps_smoothed5 = total_numsteps5[:len(smoothed_rewards5)]
    
     # Load data from the sixth text file
    total_numsteps6, rewards6 = parse_data(file_path6)
    smoothed_rewards6 = smooth_data(rewards6)
    total_numsteps_smoothed6 = total_numsteps6[:len(smoothed_rewards6)]
    
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(total_numsteps_smoothed1, smoothed_rewards1, label='HH: Smoothed Rewards', color='blue')
    plt.plot(total_numsteps_smoothed2, smoothed_rewards2, label='LIF_HH: Smoothed Rewards', color='green')
    plt.plot(total_numsteps_smoothed3, smoothed_rewards3, label='HH_modified: Smoothed Rewards', color='red')
    plt.plot(total_numsteps_smoothed4, smoothed_rewards4, label='LIF: Smoothed Rewards', color='black')
    # plt.plot(total_numsteps_smoothed5, smoothed_rewards5, label='IZH: Smoothed Rewards', color='purple')
    plt.plot(total_numsteps_smoothed6, smoothed_rewards6, label='HH_can: Smoothed Rewards', color='brown')
    
    # Set x-axis formatter to powers of 1e4
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(4, 4))
    
    ay = plt.gca()
    ay.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ay.ticklabel_format(axis='y', style='sci', scilimits=(2, 2))

    plt.xlabel('Total Number of Steps (x $10^4$)')
    plt.ylabel('Reward')
    plt.title('Total Number of Steps vs. Smoothed Reward (Two Files)')
    plt.legend()
    plt.grid()
    plt.show()

# Specify the file paths containing the data
file_path1 = 'results/result_HH.txt'  # Replace with the actual path to your first text file
file_path2 = 'results/result_LIF_HH.txt'  # Replace with the actual path to your second text file
file_path3 = 'results/result_HHup.txt'
file_path4 = 'results/result_LIF.txt'
file_path5 = 'results/result_IZH.txt'
file_path6 = 'results/result_HHcan.txt'


# Execute the script
if __name__ == "__main__":
    main(file_path1, file_path2, file_path3, file_path4, file_path5, file_path6)
