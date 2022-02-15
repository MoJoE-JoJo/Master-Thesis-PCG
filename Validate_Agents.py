import gym
from MAFGym.MAFEnv import MAFEnv
from MAFGym.MAFRandEnv import MAFRandEnv
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
        model = PPO2.load(agent_path+ "/" +checkpoint)
        step = int(checkpoint.replace("mario_","").replace(".zip",""))
        returnScore, winScore = play(num_of_val_plays, env, model)
        data.append([step, returnScore, winScore])
    header = ['Steps', 'Avg. Return', 'WinRate']
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerow(header) # 4. write the header
        csvwriter.writerows(data) # 5. write the rest of the data



levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)
env = MAFEnv(levelString, 60, False, 0, 0)

validate_agent(env, "saved_agents/new_arch/single/1e-4", 100, "new_arch;single;1e-4;lvl-1")
validate_agent(env, "saved_agents/new_arch/single/1e-5", 100, "new_arch;single;1e-5;lvl-1")
validate_agent(env, "saved_agents/less_detail/single/1e-4", 100, "less_detail;single;1e-4;lvl-1")
validate_agent(env, "saved_agents/less_detail/multiple/1e-4", 100, "less_detail;multiple;1e-4;lvl-1")
validate_agent(env, "saved_agents/less_detail/multiple/1e-5", 100, "less_detail;multiple;1e-5;lvl-1")


levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-3.txt"
levelString = readLevelFile(levelFilePath)
env.setLevel(levelString)
validate_agent(env, "saved_agents/less_detail/single/1e-4", 100, "less_detail;single;1e-4;lvl-3")
validate_agent(env, "saved_agents/less_detail/multiple/1e-4", 100, "less_detail;multiple;1e-4;lvl-3")
validate_agent(env, "saved_agents/less_detail/multiple/1e-5", 100, "less_detail;multiple;1e-5;lvl-3")