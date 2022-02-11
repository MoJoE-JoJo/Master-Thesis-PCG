import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
import os

from gym import spaces
import numpy as np
import tensorflow as tf


from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1

levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)
env = MAFEnv(levelString, 100, True)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
#env = DummyVecEnv([lambda: env])

action_space=spaces.MultiBinary(5)
observation_space = spaces.Box(low=-100, high=100, shape=(16, 16), dtype=np.uint8)
env_count = 1
steps = 3000
batch = 64
layers = [512,512]

class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            feature_extraction="mlp")

#session = tf.compat.v1.get_default_session()

#model = PPO1(MarioPolicy, env, verbose=1)
model = PPO1.load("mario_100000", env)
model.learn(total_timesteps=100000)
model.save("mario_200000")

model = PPO1.load("mario_200000", env)
model.learn(total_timesteps=100000)
model.save("mario_300000")

model = PPO1.load("mario_300000", env)
model.learn(total_timesteps=100000)
model.save("mario_400000")

model = PPO1.load("mario_400000", env)
model.learn(total_timesteps=100000)
model.save("mario_500000")

#model = PPO1.load("mario_100000")

#obs = env.reset()
#for i in range(10000):
#    action, _states = model.predict(obs)
#    obs, rewards, done, info = env.step(action)
#    env.render()
#    if done:
#        obs = env.reset()