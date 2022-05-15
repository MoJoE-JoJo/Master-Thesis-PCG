import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
import os
from time import sleep
from Level_Slicer import makeSlices
from MAFGym.MAFPCGEnv import MAFPCGEnv
from os import listdir
from os.path import isfile, join

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


#FeedForwardPolicy()
#model = PPO1(MarioPolicy, env, verbose=1)
def train(steps, saveFolder, env, learn, startNetwork = 0, num_of_checkpoints = 10, steps_per_checkpoint = 1000000, gamma = 0.99):
    if startNetwork == 0: model = PPO2(MarioPolicy, env, verbose=1, n_steps=steps, learning_rate=learn, gamma=gamma)
    else: 
        model = PPO2.load(saveFolder+"Mario_"+str(startNetwork), env)
    for i in range(num_of_checkpoints):
        file = saveFolder + "mario_" + str(startNetwork + (i+1) * steps_per_checkpoint)
        model.learn(steps_per_checkpoint)
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
        sleep(0.33)


#env2 = MAFEnv([levelString], 60, False)

#vec_env = DummyVecEnv([lambda: env1, lambda: env2])

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-3.txt"
#levelString = readLevelFile(levelFilePath)
#normal_env.setLevel(levelString)
#normal_env = MAFEnv(levelString, 100, True, 1, 1)
#play("saved_agents/vectorized/lvl_7/mario_10000000", env1)
def make_dummyVecEnv(levelStrings, rewardFunction):
    env1 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env2 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env3 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env4 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env5 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env6 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env7 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env8 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env9 = MAFEnv(levelStrings, 30, False, rewardFunction)
    env10 = MAFEnv(levelStrings, 30, False, rewardFunction)

    env_1 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])
    return env_1

mypath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\custom\\"

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

level_strings = []

for f in onlyfiles:
    level_strings.append(readLevelFile(mypath+f))

levelFilePath1 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-1.txt"
levelString1 = readLevelFile(levelFilePath1)
levelFilePath2 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-2.txt"
levelString2 = readLevelFile(levelFilePath2)
levelFilePath7 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-7.txt"
levelString7 = readLevelFile(levelFilePath7)
levelFilePath8 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-8.txt"
levelString8 = readLevelFile(levelFilePath8)

#env__1_10 = MAFEnv([levelString1], 60, False, 10)
#env__2_10 = MAFEnv([levelString2], 60, False, 10)
#env__7_10 = MAFEnv([levelString7], 60, False, 10)
#env__8_10 = MAFEnv([levelString8], 60, False, 10)

#env_rand = make_dummyVecEnv(level_strings, 10)
env_rand = MAFEnv(level_strings, 30, False, 10)

#train(512,"simplified_solver/random/", env_rand, 0.00005, 170000000, 6, 5000000, 0.99)
#validate_agent(env__2_10, "simplified_solver/random/", 100, "simplified_solver_random;lvl-2")
#validate_agent(env__7_10, "simplified_solver/random/", 100, "simplified_solver_random;lvl-7")
#validate_agent(env__8_10, "simplified_solver/random/", 100, "simplified_solver_random;lvl-8")
#validate_agent(env__1_10, "simplified_solver/random/", 100, "simplified_solver_random;lvl-1")

validate_agent(env_rand, "simplified_solver/random/", 1000, "simplified_solver_random;random")
validate_agent(env_rand, "simplified_solver/saved/", 1000, "simplified_solver;random")



#env_strange = MAFEnv([levelString1], 60, True, 10)
#play("simplified_solver/saved/mario_80000000", env_strange)
#----------------------------------------------------------------------------------------------------------
