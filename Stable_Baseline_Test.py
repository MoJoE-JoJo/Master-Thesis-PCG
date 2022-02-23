import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
import os
from time import sleep

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

def modified_cnn(scaled_images, **kwargs):
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=1024, init_scale=np.sqrt(2)))

class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp")

class MarioCnnPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioCnnPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            act_fun=tf.nn.relu,
                                            cnn_extractor=modified_cnn,
                                            feature_extraction="cnn")                                            

layers_lstm = ['lstm',dict(vf=[512,512], pi=[512,512])]

class MarioLstmPolicy(LstmPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioLstmPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers_lstm,
                                            n_lstm=256,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp")

#FeedForwardPolicy()
#model = PPO1(MarioPolicy, env, verbose=1)
def train(steps, saveFolder, env, learn, startNetwork = 0, num_of_checkpoints = 10, steps_per_checkpoint = 1000000):
    if startNetwork == 0: model = PPO2(MarioPolicy, env, verbose=1, n_steps=steps, learning_rate=learn)
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
        sleep(0.033)


#env2 = MAFEnv([levelString], 60, False)

#vec_env = DummyVecEnv([lambda: env1, lambda: env2])

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-3.txt"
#levelString = readLevelFile(levelFilePath)
#normal_env.setLevel(levelString)
#normal_env = MAFEnv(levelString, 100, True, 1, 1)
#play("saved_agents/vectorized/lvl_7/mario_10000000", env1)

levelFilePath1 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString1 = readLevelFile(levelFilePath1)
levelFilePath2 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-2.txt"
levelString2 = readLevelFile(levelFilePath2)
levelFilePath7 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
levelString7 = readLevelFile(levelFilePath7)
levelFilePath8 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-8.txt"
levelString8 = readLevelFile(levelFilePath8)

env1 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env2 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env3 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env4 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env5 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env6 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env7 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env8 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env9 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)
env10 = MAFEnv([levelString1,levelString2,levelString7,levelString8], 60, False)

env__1 = MAFEnv([levelString1], 60, False)
env__2 = MAFEnv([levelString2], 60, False)
env__7 = MAFEnv([levelString7], 60, False)
env__8 = MAFEnv([levelString8], 60, False)


env_4 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])

train(512,"saved_agents/disc_vec/mul_4/", env_4, 0.00005, 20000000, 5, 2000000)

validate_agent(env__1, "saved_agents/disc_vec/mul_4/", 100, "disc_vec;mul_4;5e-5;lvl-1")
validate_agent(env__2, "saved_agents/disc_vec/mul_4/", 100, "disc_vec;mul_4;5e-5;lvl-2")
validate_agent(env__7, "saved_agents/disc_vec/mul_4/", 100, "disc_vec;mul_4;5e-5;lvl-7")
validate_agent(env__8, "saved_agents/disc_vec/mul_4/", 100, "disc_vec;mul_4;5e-5;lvl-8")

print("mult_4 done")

#----------------------------------------------------------------------------------------------------------
"""
levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
levelString = readLevelFile(levelFilePath)
env1 = MAFEnv([levelString], 60, False)
env2 = MAFEnv([levelString], 60, False)
env3 = MAFEnv([levelString], 60, False)
env4 = MAFEnv([levelString], 60, False)
env5 = MAFEnv([levelString], 60, False)
env6 = MAFEnv([levelString], 60, False)
env7 = MAFEnv([levelString], 60, False)
env8 = MAFEnv([levelString], 60, False)
env9 = MAFEnv([levelString], 60, False)
env10 = MAFEnv([levelString], 60, False)

env_7 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])

train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 0)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 1000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 2000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 3000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 4000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 5000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 6000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 7000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 8000000)
train(512,"saved_agents/disc_vec/lvl_7/", env_7, 0.00005, 9000000)

validate_agent(env1, "saved_agents/disc_vec/lvl_7/", 100, "disc_vec;single;5e-5;lvl-7")
print("7 done")


#------------------------------------------------------------------------------------------------------------
levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)
env11 = MAFEnv([levelString], 60, False)
env12 = MAFEnv([levelString], 60, False)
env13 = MAFEnv([levelString], 60, False)
env14 = MAFEnv([levelString], 60, False)
env15 = MAFEnv([levelString], 60, False)
env16 = MAFEnv([levelString], 60, False)
env17 = MAFEnv([levelString], 60, False)
env18 = MAFEnv([levelString], 60, False)
env19 = MAFEnv([levelString], 60, False)
env20 = MAFEnv([levelString], 60, False)

env_1 = DummyVecEnv([lambda: env11,lambda: env12,lambda: env13,lambda: env14,lambda: env15,lambda: env16,lambda: env17,lambda: env18,lambda: env19,lambda: env20,])

train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 0)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 1000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 2000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 3000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 4000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 5000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 6000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 7000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 8000000)
train(512,"saved_agents/disc_vec/lvl_1/", env_1, 0.00005, 9000000)

validate_agent(env11, "saved_agents/disc_vec/lvl_1/", 100, "disc_vec;single;5e-5;lvl-1")
print("1 done")


#------------------------------------------------------------------------------------------------------------
levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-2.txt"
levelString = readLevelFile(levelFilePath)
env21 = MAFEnv([levelString], 60, False)
env22 = MAFEnv([levelString], 60, False)
env23 = MAFEnv([levelString], 60, False)
env24 = MAFEnv([levelString], 60, False)
env25 = MAFEnv([levelString], 60, False)
env26 = MAFEnv([levelString], 60, False)
env27 = MAFEnv([levelString], 60, False)
env28 = MAFEnv([levelString], 60, False)
env29 = MAFEnv([levelString], 60, False)
env30 = MAFEnv([levelString], 60, False)

env_2 = DummyVecEnv([lambda: env21,lambda: env22,lambda: env23,lambda: env24,lambda: env25,lambda: env26,lambda: env27,lambda: env28,lambda: env29,lambda: env30,])

train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 0)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 1000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 2000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 3000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 4000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 5000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 6000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 7000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 8000000)
train(512,"saved_agents/disc_vec/lvl_2/", env_2, 0.00005, 9000000)

validate_agent(env21, "saved_agents/disc_vec/lvl_2/", 100, "disc_vec;single;5e-5;lvl-2")
print("2 done")

#------------------------------------------------------------------------------------------------------------
levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-8.txt"
levelString = readLevelFile(levelFilePath)
env31 = MAFEnv([levelString], 60, False)
env32 = MAFEnv([levelString], 60, False)
env33 = MAFEnv([levelString], 60, False)
env34 = MAFEnv([levelString], 60, False)
env35 = MAFEnv([levelString], 60, False)
env36 = MAFEnv([levelString], 60, False)
env37 = MAFEnv([levelString], 60, False)
env38 = MAFEnv([levelString], 60, False)
env39 = MAFEnv([levelString], 60, False)
env40 = MAFEnv([levelString], 60, False)

env_3 = DummyVecEnv([lambda: env31,lambda: env32,lambda: env33,lambda: env34,lambda: env35,lambda: env36,lambda: env37,lambda: env38,lambda: env39,lambda: env40,])

train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 0)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 1000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 2000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 3000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 4000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 5000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 6000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 7000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 8000000)
train(512,"saved_agents/disc_vec/lvl_8/", env_3, 0.00005, 9000000)

validate_agent(env31, "saved_agents/disc_vec/lvl_8/", 100, "disc_vec;single;5e-5;lvl-8")
print("8 done")

"""