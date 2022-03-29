import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
import os
from time import sleep
from Level_Slicer import makeSlices
from MAFGym.MAFPCGEnv import MAFPCGEnv

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
        sleep(0.033)


#env2 = MAFEnv([levelString], 60, False)

#vec_env = DummyVecEnv([lambda: env1, lambda: env2])

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-3.txt"
#levelString = readLevelFile(levelFilePath)
#normal_env.setLevel(levelString)
#normal_env = MAFEnv(levelString, 100, True, 1, 1)
#play("saved_agents/vectorized/lvl_7/mario_10000000", env1)
def make_dummyVecEnv(levelStrings, rewardFunction):
    env1 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env2 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env3 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env4 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env5 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env6 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env7 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env8 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env9 = MAFEnv(levelStrings, 60, False, rewardFunction)
    env10 = MAFEnv(levelStrings, 60, False, rewardFunction)

    env_1 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])
    return env_1


levelFilePath1 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString1 = readLevelFile(levelFilePath1)
levelFilePath2 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-2.txt"
levelString2 = readLevelFile(levelFilePath2)
levelFilePath7 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
levelString7 = readLevelFile(levelFilePath7)
levelFilePath8 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-8.txt"
levelString8 = readLevelFile(levelFilePath8)

#env__9 = MAFEnv([levelString7], 60, False, 9)
#env__10 = MAFEnv([levelString1], 60, False, 10)
#env__11 = MAFEnv([levelString1], 60, False, 11)
#env__12 = MAFEnv([levelString1], 60, False, 12)
#env__13 = MAFEnv([levelString1], 60, False, 13)
#env__14 = MAFEnv([levelString1], 60, False, 14)

#env__1_10 = MAFEnv([levelString1], 60, False, 10)
#env__2_10 = MAFEnv([levelString2], 60, False, 10)
#env__7_10 = MAFEnv([levelString7], 60, False, 10)
#env__8_10 = MAFEnv([levelString8], 60, False, 10)


#env__1_4 = MAFEnv([levelString1], 60, False, 4)
#env__2_4 = MAFEnv([levelString2], 60, False, 4)
#env__7_4 = MAFEnv([levelString7], 60, False, 4)
#env__8_4 = MAFEnv([levelString8], 60, False, 4)


#env_mul_10 = make_dummyVecEnv([levelString1,levelString2,levelString7,levelString8], 10)
#env_mul_4 = make_dummyVecEnv([levelString1,levelString2,levelString7,levelString8], 4)


#train(512,"saved_agents/rew_shap/mult/10/", env_mul_10 , 0.00005, 40000000, 30, 2000000, 0.99)
#validate_agent(env__1_10, "saved_agents/rew_shap/mult/10/", 100, "rew_shap_10_2;mul;5e-5;lvl-1")
#validate_agent(env__2_10, "saved_agents/rew_shap/mult/10/", 100, "rew_shap_10_2;mul;5e-5;lvl-2")
#validate_agent(env__7_10, "saved_agents/rew_shap/mult/10/", 100, "rew_shap_10_2;mul;5e-5;lvl-7")
#validate_agent(env__8_10, "saved_agents/rew_shap/mult/10/", 100, "rew_shap_10_2;mul;5e-5;lvl-8")

#train(512,"saved_agents/rew_shap/mult/4/", env_mul_4 , 0.00005, 20000000, 10, 2000000, 0.99)
#validate_agent(env__1_4, "saved_agents/rew_shap/mult/4/", 100, "rew_shap_4;mul;5e-5;lvl-1")
#validate_agent(env__2_4, "saved_agents/rew_shap/mult/4/", 100, "rew_shap_4;mul;5e-5;lvl-2")
#validate_agent(env__7_4, "saved_agents/rew_shap/mult/4/", 100, "rew_shap_4;mul;5e-5;lvl-7")
#validate_agent(env__8_4, "saved_agents/rew_shap/mult/4/", 100, "rew_shap_4;mul;5e-5;lvl-8")

#play("saved_agents/rew_shap/mult/4/mario_30000000", env__2_4)
#play("saved_agents/disc_vec/mul_4/mario_30000000", env__2_4)
#play("saved_agents/rew_shap/mult/10/mario_30000000", env__2_4)


level_folder ="MAFGym/levels/original/subset/completable"

slices = makeSlices(level_folder)

generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"
#env = MAFPCGEnv(0, slices, generated_level_path)

#slices = []
#slices.append(env.slice_map[5])
##slices.append(env.slice_map[74])
#slices.append(env.slice_map[188])
#slices.append(env.slice_map[360])
#slices.append(env.slice_map[464])


lines = [""] * 16
for slice in slices:
    for line_index in range(len(slice)):
        lines[line_index] += slice[line_index]

levelString = ""
for line in lines:
    levelString += line 
    levelString += "\n"

env_strange = MAFEnv([levelString8], 60, True, 10)
play("saved_agents/rew_shap/mult/10/mario_100000000", env_strange)
#----------------------------------------------------------------------------------------------------------
