import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
import os
from time import sleep

from gym import spaces
import numpy as np
import tensorflow as tf


from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1

#env = make_vec_env('CartPole-v1', n_envs=4)
#obs = env.reset()
levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)
env = MAFEnv(levelString, 100, True)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
#env = DummyVecEnv([lambda: env])

#action_space=spaces.MultiBinary(5)
#observation_space = spaces.Box(low=-100, high=100, shape=(16, 16), dtype=np.uint8)
env_count = 1
#steps = 3000
#batch = 64
layers = [512,512]

class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            feature_extraction="mlp")

#model = PPO1(MarioPolicy, env, verbose=1)

def train():
    num_of_steps = 100000
    model = PPO2(MarioPolicy, env, verbose=1, n_steps=512, learning_rate=0.00001)
    for i in range(10):
        file = "saved_agents/mario_" + str((i+1)*num_of_steps) + "_2"
        model.learn(num_of_steps)
        model.save(file)

def play():
    model = PPO2.load("saved_agents/mario_100000_2")
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        sleep(0.033)


train()