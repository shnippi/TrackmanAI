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

# TODO: implement if <-0.5 if presses w and s together to break and compare this with only break
# TODO: PULL REQUEST
# TODO: YOUTUBE VIDEO?
# TODO: OCR THE MINUS SIGN BEFORE SPEED, THEN USE THIS AS RESET CONDITION
# TODO: speed still gets read incorrectly
# TODO: sometimes reward is in the millions????


load_dotenv()
wandb.login()
wandb.init(project="Trackmania")

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

hyperparameters = dict(
    gamma=args.gamma,
    tau=args.tau,
    lr=args.lr,
    alpha=args.alpha,
    batch_size=args.batch_size,
    num_steps=args.num_steps,
    hidden_size=args.hidden_size,
    start_steps=args.start_steps,
    policy=args.policy,
    env_name=args.env_name,
    eval=args.eval,
    automatic_entropy_tuning=args.automatic_entropy_tuning,
    seed=args.seed,
    updates_per_step=args.updates_per_step,
    target_update_interval=args.target_update_interval,
    replay_size=args.replay_size,
    cuda=args.cuda,
)

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

agent.load_checkpoint("checkpoints/sac_checkpoint_Trackmania_")
memory.load_buffer("checkpoints/sac_buffer_Trackmania_")

print(len(memory.buffer))

# tensorboard --logdir=SAC/runs --port=6006
writer = SummaryWriter(
    'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                  args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Training Loop
total_numsteps = 0
updates = 0
best_reward = 0
load_from_best_cntr = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:

        if args.start_steps > total_numsteps:
            action = env.random_action()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
            # print(action)

        # press P to pause
        keys = key_check()
        if 'P' in keys:
            print("PAUSED")
            while True:
                time.sleep(1)
                keys = key_check()

                if 'P' in keys:
                    print("UNPAUSED")
                    break

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networkdw
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)
                wandb.log({"critic_1_loss": critic_1_loss})
                wandb.log({"critic_2_loss": critic_2_loss})
                wandb.log({"policy_loss": policy_loss})
                wandb.log({"ent_loss": ent_loss})
                wandb.log({"alpha": alpha})

                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # print(reward)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        # mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        mask = 1
        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state

        # print(total_numsteps)
        # if episode_steps > 130:
        #     done = True

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))

    if i_episode % 10 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 3
        for _ in range(episodes):
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

            avg_reward += episode_reward
        avg_reward /= episodes

        wandb.log({"avg_reward": avg_reward})
        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

        if avg_reward > best_reward:
            agent.save_checkpoint("Trackmania", suffix="best")
            best_reward = avg_reward
            load_from_best_cntr = 0
        else:
            load_from_best_cntr += 1

        # reset after an amount of episodes that it didnt reach a new highscore anymore
        if load_from_best_cntr > 15:
            agent.load_checkpoint("checkpoints/sac_checkpoint_Trackmania_best")
            load_from_best_cntr = 0

        agent.save_checkpoint("Trackmania")
        memory.save_buffer("Trackmania")
 