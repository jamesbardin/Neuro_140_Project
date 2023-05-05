import gym
import torch
import torch.nn as nn
import numpy as np
from lib import wrappers
from lib import wrappers_skips
from lib import dqn_model
import collections
import argparse
import time
import cv2
import gym
import gym.spaces
import collections


# envs = [wrappers_testing.make_env("PongNoFrameskip-v4", seed) for seed in range(1)]

env = wrappers_skips.make_env("PongNoFrameskip-v4", 7)
env.seed(42)
net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
net.load_state_dict(torch.load('C:/NEURO140/Final/PongNoFrameskip-v4-skip7.dat', map_location=lambda storage, loc: storage))

state = env.reset()
total_reward = 0.0
c = collections.Counter()

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('seedlele5.mp4', fourcc, 60.0, (env.render(mode='rgb_array').shape[1], env.render(mode='rgb_array').shape[0]))

done = False
obs = env.reset()
while not done:
    action = net(torch.tensor([obs])).argmax(dim=1).item()
    obs, reward, done, _ = env.step(action)
    out.write(env.render(mode='rgb_array'))

# Release everything if job is finished
out.release()
cv2.destroyAllWindows()




