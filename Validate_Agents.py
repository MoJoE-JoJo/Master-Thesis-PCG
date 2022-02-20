import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
import os
from time import sleep
import csv

from os import listdir
from os.path import isfile, join

from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow import Tensor

from stable_baselines.common.policies import CnnPolicy, FeedForwardPolicy
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, PPO1


def play(num_of_val_plays, env, model):
    totalReturn = 0
    totalWins = 0
    for i in range(num_of_val_plays):
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)
            if done:
                totalReturn += float(info.get("ReturnScore"))
                if info.get("Result") == "Win": totalWins += 1
                if info.get("Result") == "Lose": totalWins += 0
                obs = env.reset()
    return totalReturn/num_of_val_plays, totalWins



def validate_agent(env, agent_path, num_of_val_plays, saveName):
    allCheckpoints = [f for f in listdir(agent_path) if isfile(join(agent_path, f))]
    filename = "agent_validations/" + saveName + ".csv"
    data = []
    for checkpoint in allCheckpoints:
        model = PPO2.load(agent_path +checkpoint)
        step = int(checkpoint.replace("mario_","").replace(".zip",""))
        returnScore, winScore = play(num_of_val_plays, env, model)
        data.append([step, returnScore, winScore])
    header = ['Steps', 'Avg. Return', 'WinRate']
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerow(header) # 4. write the header
        csvwriter.writerows(data) # 5. write the rest of the data


levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
levelString = readLevelFile(levelFilePath)
env = MAFEnv([levelString], 60, False)

levelFilePath1 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString1 = readLevelFile(levelFilePath1)
env1 = MAFEnv([levelString1], 60, False)


validate_agent(env, "saved_agents/basic_network/lvl_7/", 100, "basic;single;5e-5;lvl-7")
print("7 done")
validate_agent(env1, "saved_agents/basic_nework/lvl_1/", 100, "basic;single;5e-5;lvl-1")
print("1 done")

#env = MAFEnv(levelString, 30, False, 1, 1)
#validate_agent(env, "saved_agents/less_detail/single/1e-4", 100, "less_detail;single;1e-4;lvl-1")
#print("3 done")
#validate_agent(env, "saved_agents/less_detail/multiple/1e-4", 100, "less_detail;multiple;1e-4;lvl-1")
#print("4 done")
#validate_agent(env, "saved_agents/less_detail/multiple/1e-5", 100, "less_detail;multiple;1e-5;lvl-1")
#print("5 done")


#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
#levelString = readLevelFile(levelFilePath)
#env = MAFEnv([levelString], 60, False)
#validate_agent(env, "saved_agents/old_new_env/new_env/lvl_1/", 100, "new_env;5e-5;lvl_1")
#print("1 done")

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-2.txt"
#levelString = readLevelFile(levelFilePath)
#env = MAFEnv([levelString], 60, False)
#validate_agent(env, "saved_agents/old_new_env/new_env/lvl_2/", 100, "new_env;5e-5;lvl_2")
#print("2 done")

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-7.txt"
#levelString = readLevelFile(levelFilePath)
#env = MAFEnv([levelString], 60, False)
#validate_agent(env, "saved_agents/old_new_env/new_env/lvl_7/", 100, "new_env;5e-5;lvl_7")
#print("7 done")

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-8.txt"
#levelString = readLevelFile(levelFilePath)
#env = MAFEnv([levelString], 60, False)
#validate_agent(env, "saved_agents/old_new_env/new_env/lvl_8/", 100, "new_env;5e-5;lvl_8")
#print("8 done")

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-11.txt"
#levelString = readLevelFile(levelFilePath)
#env = MAFEnv([levelString], 60, False)
#validate_agent(env, "saved_agents/old_new_env/new_env/lvl_11/", 100, "new_env;5e-5;lvl_11")
#print("11 done")

#levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-3.txt"
#levelString = readLevelFile(levelFilePath)
#env = MAFEnv([levelString], 60, False)
#validate_agent(env, "saved_agents/old_new_env/new_env/lvl_3/", 100, "new_env;5e-5;lvl_3")
#print("3 done")