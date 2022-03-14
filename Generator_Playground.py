import gym
from MAFGym.MAFPCGEnv import MAFPCGEnv
from MAFGym.util import readLevelFile
import os
from time import sleep
from Level_Slicer import makeSlices
from Generator import Generator

from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy, LstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1

from Validate_Agents import validate_agent

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
#layers = [512,512,512,512]
layers = [dict(vf=[512,512], pi=[512,512])]

class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp")                                         

def train(saveFolder, generator: Generator, num_of_checkpoints = 10, steps_per_checkpoint = 1000000):
    model = generator.model
    for i in range(num_of_checkpoints):
        model.learn(steps_per_checkpoint)
        generator.increment_steps_trained(steps_per_checkpoint)
        generator.save(saveFolder)



level_folder ="MAFGym/levels/original/subset/completable/lvl-1"

generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"
generator = Generator(load_path=os.getcwd()+"\\saved_pcg\\simple3\\pcg_50000.zip",
    levels_path=level_folder, 
    generate_path=generated_level_path, 
    steps=32, 
    learn=0.00005,
    gamma=0.99)


#generator.load(os.getcwd()+"\\saved_pcg\\simple3\\pcg_50000.zip")

obs = generator.env.reset()
for i in range(100):
    action, _states = generator.model.predict(obs)
    obs, rewards, done, info = generator.env.step(action)
    if done:
        obs = generator.env.reset()

#----------------------------------------------------------------------------------------------------------
