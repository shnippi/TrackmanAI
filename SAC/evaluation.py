# Original Implementation taken from: https://github.com/pranz24/pytorch-soft-actor-critic

import argparse
import datetime
import os
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from Trackmania_env import Trackmania_env
import time
from getkeys import key_check
import wandb
from dotenv import load_dotenv
import msvcrt as m

# load_dotenv()
# wandb.login()
# wandb.init(project="Trackmania")

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="Trackmania",
                    help='IDK what you want help with lol')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=-1, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=True,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

print("Random start steps: " + str(args.start_steps))
for i in list(range(3))[::-1]:
    print(i + 1)
    time.sleep(1)

# Environment
env = Trackmania_env()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

agent.load_checkpoint("checkpoints/sac_checkpoint_Trackmania_best_stable", evaluate=True)
# memory.load_buffer("checkpoints/sac_buffer_Trackmania_stable")

print(len(memory.buffer))

for i_episode in itertools.count(1):
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(state, evaluate=True)
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        state = next_state

        keys = key_check()
        if 'P' in keys:
            print("PAUSED")
            while True:
                time.sleep(1)
                keys = key_check()

                if 'P' in keys:
                    print("UNPAUSED")
                    break
    # wandb.log({"avg_reward": episode_reward})

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(i_episode, round(episode_reward, 2)))
    print("----------------------------------------")
