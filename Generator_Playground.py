import gym
from MAFGym.MAFPCGEnv import MAFPCGEnv
from MAFGym.util import readLevelFile
import os
from time import sleep
from Level_Slicer import makeSlices

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
def train(steps, saveFolder, env, learn, startNetwork = 0, num_of_checkpoints = 10, steps_per_checkpoint = 1000000, gamma = 0.99):
    if startNetwork == 0: model = PPO2(MarioPolicy, env, verbose=1, n_steps=steps, learning_rate=learn, gamma=gamma)
    else: 
        model = PPO2.load(saveFolder+"PCG_"+str(startNetwork), env)
    for i in range(num_of_checkpoints):
        file = saveFolder + "PCG_" + str(startNetwork + (i+1) * steps_per_checkpoint)
        model.learn(steps_per_checkpoint)
        model.save(file)


level_folder ="MAFGym/levels/original/subset/completable/lvl-1"

slices = makeSlices(level_folder)

generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"
env = MAFPCGEnv(0, slices, generated_level_path)

train(32,"saved_pcg/simple/", env , 0.00005, 0, 10, 1000000, 0.99)

#----------------------------------------------------------------------------------------------------------
