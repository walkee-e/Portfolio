import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from matplotlib import animation
import matplotlib.pyplot as plt
from replay_memory import ReplayMemory

# Argument Parsing
parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="LunarLanderContinuous-v2",
                    help='Gym environment (default: LunarLanderContinuous-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy every 10 episodes (default: False)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--num_steps', type=int, default=101100, metavar='N',
                    help='maximum number of steps (default: 101100)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='hidden size (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=2000, metavar='N',
                    help='Steps sampling random actions (default: 2000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--model_name', default="LIF_HH",
                    help='choose model (choice: LIF, HH, LIF_HH, 4LIF, ANN)')
args = parser.parse_args()

# Main Training Function
def train(args):
    # Environment
    env = gym.make(args.env_name)
    env.seed(args.seed)
    env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Agent
    if args.model_name == "LIF":
        from models.model_lif import GaussianPolicy
    elif args.model_name == "HH":
        from models.model_hh import GaussianPolicy
    elif args.model_name == "LIF_HH":
        from models.model_lif_hh import GaussianPolicy
    elif args.model_name == "HH_CAN":
        from models.model_hh_can import GaussianPolicy
    elif args.model_name == "HH_MODIFIED":
        from models.model_hh_modified import GaussianPolicy
    elif args.model_name == "IZH":
        from models.model_izh import GaussianPolicy    
    else:
        raise ValueError(f"Unknown model name: {args.model_name}")

    agent = SAC(env.observation_space.shape[0], env.action_space, args, GaussianPolicy)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    negative_reward = -10.0
    wins = 5  # Window size for state sequences

    reward_dict_iteration = []

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()
        state_dict = []
        action_dict = []
        reward_dict = []
        mask_dict = []
        next_state_dict = []

        while not done:
            steps = len(state_dict)
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                if steps < wins:  # If not enough states in memory
                    state_tmp = torch.zeros((wins, state.shape[0]), device=agent.device)  # Initialize with zeros
                else:
                    state_tmp = torch.stack([state_dict[steps - wins + i] for i in range(wins)])

                action = agent.select_action(state_tmp)

            if len(memory) > args.batch_size and not args.eval:
                for _ in range(args.updates_per_step):
                    # Update parameters of all the networks
                    agent.update_parameters(memory, args.batch_size, updates)
                    updates += 1

            next_state, reward, done, _ = env.step(action)
            if done:
                reward += negative_reward

            total_numsteps += 1
            episode_reward += reward
            mask = 1 if episode_steps == env._max_episode_steps else float(not done)

            # Save transition to memory
            state_dict.append(torch.FloatTensor(state))
            action_dict.append(torch.tensor(action))
            reward_dict.append(torch.tensor(reward).view(1, -1))
            next_state_dict.append(torch.tensor(next_state))
            mask_dict.append(torch.tensor(mask).view(1, -1))

            steps = len(state_dict)
            if steps >= wins:
                state_dict_tmp = torch.stack([state_dict[steps - wins + i] for i in range(wins)])
                action_dict_tmp = torch.stack([action_dict[steps - wins + i] for i in range(wins)])
                reward_dict_tmp = torch.stack([reward_dict[steps - wins + i] for i in range(wins)], dim=1)
                next_state_dict_tmp = torch.stack([next_state_dict[steps - wins + i] for i in range(wins)])
                mask_dict_tmp = torch.stack([mask_dict[steps - wins + i] for i in range(wins)], dim=1)
                memory.push(state_dict_tmp, action_dict_tmp, reward_dict_tmp, next_state_dict_tmp, mask_dict_tmp)

            state = next_state


        print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, "
              f"episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

        if total_numsteps >= args.num_steps:
            break

    env.close()

# Run Training
train(args)
