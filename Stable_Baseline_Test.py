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
def train(steps, saveFolder, env, learn, startNetwork):
    num_of_steps = 1000000
    num_of_times = 1
    if startNetwork == 0: model = PPO2(MarioPolicy, env, verbose=1, n_steps=steps, learning_rate=learn)
    else: 
        model = PPO2.load(saveFolder+"Mario_"+str(startNetwork), env)
    for i in range(num_of_times):
        file = saveFolder + "mario_" + str(startNetwork + (i+1) * num_of_steps)
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


levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
levelString = readLevelFile(levelFilePath)
env1 = MAFEnv([levelString], 60, True)
env2 = MAFEnv([levelString], 60, False)
env3 = MAFEnv([levelString], 60, False)
env4 = MAFEnv([levelString], 60, False)
env5 = MAFEnv([levelString], 60, False)
env6 = MAFEnv([levelString], 60, False)
env7 = MAFEnv([levelString], 60, False)
env8 = MAFEnv([levelString], 60, False)
env9 = MAFEnv([levelString], 60, False)
env10 = MAFEnv([levelString], 60, False)

env = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])
#env2 = MAFEnv([levelString], 60, False)

#vec_env = DummyVecEnv([lambda: env1, lambda: env2])

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-3.txt"
#levelString = readLevelFile(levelFilePath)
#normal_env.setLevel(levelString)
#normal_env = MAFEnv(levelString, 100, True, 1, 1)
#play("saved_agents/vectorized/lvl_7/mario_10000000", env1)



train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 0)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 1000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 2000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 3000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 4000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 5000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 6000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 7000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 8000000)
train(512,"saved_agents/disc_vec/lvl_7/", env, 0.00005, 9000000)
#train(512,"saved_agents/norm_vec/lvl_7/", env, 0.00005, 10000000)
#train(512,"saved_agents/norm_vec/lvl_7/", env, 0.00005, 11000000)

#train(512,"saved_agents/vectorized/lvl_7/", env, 0.00005, 12000000)
#train(512,"saved_agents/vectorized/lvl_7/", env, 0.00005, 13000000)
#train(512,"saved_agents/vectorized/lvl_7/", env, 0.00005, 14000000)

validate_agent(env1, "saved_agents/disc_vec/lvl_7/", 100, "disc_vec;single;5e-5;lvl-7")
print("7 done")


#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
#levelString = readLevelFile(levelFilePath)
#env2 = MAFEnv([levelString], 60, False)
#env = DummyVecEnv([lambda: env2])
#train(512,"saved_agents/steps/lvl_7/", env, 0.00005, 0)
#train(512,"saved_agents/steps/lvl_7/", env, 0.00005, 1000000)
#train(512,"saved_agents/steps/lvl_7/", env, 0.00005, 2000000)
#train(512,"saved_agents/steps/lvl_7/", env, 0.00005, 3000000)
#train(512,"saved_agents/steps/lvl_7/", env, 0.00005, 4000000)
#validate_agent(env2, "saved_agents/steps/lvl_7/", 100, "steps_512;single;5e-5;lvl-7")
#print("7 done")


#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
#levelString = readLevelFile(levelFilePath)
#env3 = MAFEnv([levelString], 60, False)
#env = DummyVecEnv([lambda: env3])
#train(256,"saved_agents/cnn/lvl_7/", env, 0.00005, 0)
#train(256,"saved_agents/cnn/lvl_7/", env, 0.00005, 1000000)
#train(256,"saved_agents/cnn/lvl_7/", env, 0.00005, 2000000)
#train(256,"saved_agents/cnn/lvl_7/", env, 0.00005, 3000000)
#train(256,"saved_agents/cnn/lvl_7/", env, 0.00005, 4000000)

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-8.txt"
#levelString = readLevelFile(levelFilePath)
#env4 = MAFEnv([levelString], 60, False)
#env = DummyVecEnv([lambda: env4])
#train(256,"saved_agents/cnn/lvl_8/", env, 0.00005, 0)
#train(256,"saved_agents/cnn/lvl_8/", env, 0.00005, 1000000)
#train(256,"saved_agents/cnn/lvl_8/", env, 0.00005, 2000000)
#train(256,"saved_agents/cnn/lvl_8/", env, 0.00005, 3000000)
#train(256,"saved_agents/cnn/lvl_8/", env, 0.00005, 4000000)