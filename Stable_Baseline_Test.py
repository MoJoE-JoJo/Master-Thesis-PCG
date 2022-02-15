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

from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy
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
#layers = [512,512,512,512]
layers = [dict(vf=[512,512], pi=[512,512])]

class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            act_fun=tf.nn.relu,
                                            feature_extraction="mlp")
#FeedForwardPolicy()
#model = PPO1(MarioPolicy, env, verbose=1)
def train(steps, saveFolder, env, learn, startNetwork):
    num_of_steps = 100000
    num_of_times = 10
    if startNetwork == 0: model = PPO2(MarioPolicy, env, verbose=1, n_steps=steps, learning_rate=learn)
    else: 
        env = DummyVecEnv([lambda: env])
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


levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)
#normal_env = MAFEnv(levelString, 100, True, 0, 0)
#random_env = MAFRandEnv(100, True, 0, 0)

#play("saved_agents/new_arch/single/1e-5/mario_3000000", normal_env)


#train(256,"saved_agents/new_arch/single/1e-6/", normal_env, 0.000001, 0)
#train(256,"saved_agents/new_arch/single/1e-6/", normal_env, 0.000001, 1000000)
#train(256,"saved_agents/new_arch/single/1e-6/", normal_env, 0.000001, 2000000)

#train(256,"saved_agents/new_arch/multiple/1e-4/", random_env, 0.0001, 0)
#train(256,"saved_agents/new_arch/multiple/1e-4/", random_env, 0.0001, 1000000)
#train(256,"saved_agents/new_arch/multiple/1e-4/", random_env, 0.0001, 2000000)

#train(256,"saved_agents/new_arch/multiple/1e-6/", random_env, 0.000001, 0)
#train(256,"saved_agents/new_arch/multiple/1e-6/", random_env, 0.000001, 1000000)
#train(256,"saved_agents/new_arch/multiple/1e-6/", random_env, 0.000001, 2000000)

#train(256,"saved_agents/new_arch/single/1e-6/", normal_env, 0.000001, 3000000)
#train(256,"saved_agents/new_arch/multiple/1e-4/", random_env, 0.0001, 3000000)
#train(256,"saved_agents/new_arch/multiple/1e-6/", random_env, 0.000001, 3000000)

#train(256,"saved_agents/new_arch/single/1e-6/", normal_env, 0.000001, 4000000)
#train(256,"saved_agents/new_arch/multiple/1e-4/", random_env, 0.0001, 4000000)
#train(256,"saved_agents/new_arch/multiple/1e-6/", random_env, 0.000001, 4000000)


ls_norm_env = MAFEnv(levelString, 100, True, 1, 1)
train(256,"saved_agents/less_detail/single/1e-4/", ls_norm_env, 0.0001, 4000000)
train(256,"saved_agents/less_detail/single/1e-5/", ls_norm_env, 0.00001, 0)
train(256,"saved_agents/less_detail/single/1e-5/", ls_norm_env, 0.00001, 1000000)
train(256,"saved_agents/less_detail/single/1e-5/", ls_norm_env, 0.00001, 2000000)
train(256,"saved_agents/less_detail/single/1e-5/", ls_norm_env, 0.00001, 3000000)
train(256,"saved_agents/less_detail/single/1e-5/", ls_norm_env, 0.00001, 4000000)


ls_rand_env = MAFRandEnv(100, True, 1, 1)
train(256,"saved_agents/less_detail/multiple/1e-4/", ls_rand_env, 0.0001, 4000000)
train(256,"saved_agents/less_detail/multiple/1e-5/", ls_rand_env, 0.00001, 0)
train(256,"saved_agents/less_detail/multiple/1e-5/", ls_rand_env, 0.00001, 1000000)
train(256,"saved_agents/less_detail/multiple/1e-5/", ls_rand_env, 0.00001, 2000000)
train(256,"saved_agents/less_detail/multiple/1e-5/", ls_rand_env, 0.00001, 3000000)
train(256,"saved_agents/less_detail/multiple/1e-5/", ls_rand_env, 0.00001, 4000000)

#train(256,"saved_agents/less_detail/single/1e-4/", ls_norm_env, 0.0001, 3000000)
#train(256,"saved_agents/less_detail/multiple/1e-4/", ls_rand_env, 0.0001, 3000000)



#train(256,"saved_agents/multiple_levels/", random_env, 0.00025)
#train(256,"saved_agents/single_level_learn_0_0001/", normal_env, 0.0001)
#train(256,"saved_agents/multiple_levels_learn_0_0001/", random_env, 0.0001)

#train(512,"saved_agents/single_level_512/", normal_env, 0.00025)
#train(512,"saved_agents/multiple_levels_512/", random_env, 0.00025)
#train(512,"saved_agents/single_level_512_learn_0_0001/", normal_env, 0.0001)
#train(512,"saved_agents/multiple_levels_512_learn_0_0001/", random_env, 0.0001)