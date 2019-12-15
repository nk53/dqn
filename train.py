#!/usr/bin/env python3
#import keras
import os

folder = "data"
logfile = os.path.join(folder, "logfile.txt")
calc_rolling_avg = True
rolling_avgs_file = os.path.join(folder, "rolling_avgs.txt")

if not os.path.exists(folder):
    msg = "Output directory '{}' does not exist. Creating it ..."
    msg = msg.format(folder)
    print(msg)
    os.mkdir(folder)
elif os.path.isfile(folder):
    import sys
    msg = "Error: output directory '{}' already exists and is a regular file."
    msg = msg.format(folder)
    msg += " Please remove or rename it."
    print(msg, file=sys.stderr)
    sys.exit(1)

import gym
import DQN
import GymTrainer as gt
import importlib 
import tensorflow as tf
from Wrappers import *
from gym import wrappers
from resize_observation_keep_dims import ResizeObservationKeepDims

env_name = 'CartPole-v0'
#env_name = 'StarGunnerNoFrameskip-v4'

# setup
env = gym.make(env_name)
if 'StarGunner' in env_name:
    use_conv = True
    env = gym.wrappers.AtariPreprocessing(env, frame_skip=4,
            grayscale_obs=False, screen_size=84)
else:
    use_conv = False

if 'StarGunner' in env_name:
    #env = wrappers.EpisodicLifeEnv(env)
    env = wrappers.GrayScaleObservation(env) 
    env = ResizeObservationKeepDims(env, (84, 84))

env = wrappers.FrameStack(env, 4, lz4_compress=True)

if calc_rolling_avg:
    env = RollingMeanReturn(env, window=100)
    env = RecordInfo(env, rolling_avgs_file, ["episode"], overwritefile=False)

# try to resume training, or start new training example
try:
    model = DQN.DQNmodel(env.action_space.n, folder, {"minutes":1},
            use_convolutions=use_conv)
    replay_mem_file = model.load(backup_folder=folder)
    assert replay_mem_file
    print("-----resuming training-----")

except FileNotFoundError:
    model = DQN.DQNmodel(env.action_space.n, folder, {"minutes":1},
            use_convolutions=use_conv)
    print("-----starting training-----")
    replay_mem_file = ""

replay_start_size = int(5E4)
trainer = gt.GymTrainer(env, replay_memory_size=int(1E6), 
                    agent_history_size=4, action_repeat=4,
                    logfile=logfile, pkfile=replay_mem_file)

replay_mem_pop = len(trainer.replay_memory) 
if replay_mem_pop:
  print ("Loaded", replay_mem_pop, "memories.")

# if insufficient number of replay memories were loaded, add more.
if len(trainer.replay_memory) < replay_start_size:
    needs_memories = replay_start_size - replay_mem_pop
    needs_steps = needs_memories * trainer.action_repeat
    print("Replay-memory needs", needs_memories, "more memories.")

    # populate the replay memory with random action transitions 
    trainer.train(num_steps=needs_steps, verbose=False)

    print("Done pretraining, beginning regular training")

trainer.train(num_steps=1E6,
        policy=model.policy,
        observe=model._update,
        verbose=False)

print("Done training")

