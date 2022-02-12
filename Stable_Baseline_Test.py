import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.MAFRandEnv import MAFRandEnv
from MAFGym.util import readLevelFile
import os
from time import sleep

from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1

#env = make_vec_env('CartPole-v1', n_envs=4)
#obs = env.reset()
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
                                            act_fun=tf.nn.relu,
                                            feature_extraction="mlp")
#FeedForwardPolicy()
#model = PPO1(MarioPolicy, env, verbose=1)
def train(steps, saveFolder, env):
    num_of_steps = 1000
    num_of_times = 1
    model = PPO2(MarioPolicy, env, verbose=1, n_steps=steps, learning_rate=0.00025)
    for i in range(num_of_times):
        file = saveFolder + "mario_" + str((i+1)*num_of_steps)
        model.learn(num_of_steps)
        model.save(file)

def play(path, env):
    model = PPO2.load(path)
    obs = env.reset()
    for i in range(10000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        sleep(0.033)


levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)
normal_env = MAFEnv(levelString, 100, True)
random_env = MAFRandEnv(100, True)

train(256,"saved_agents/single_level/", normal_env)
train(256,"saved_agents/multiple_levels/", random_env)
#play("saved_agents/mario_100000_2")