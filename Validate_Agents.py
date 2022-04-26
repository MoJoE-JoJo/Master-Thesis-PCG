import gym
from ARLPCG import ARLPCG
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
from ARLPCG import ARLPCG


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


def sort_csv_data(file_name):
    with open(file_name, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        rows = []
        header = []
        counter = 0
        for row in reader:
            if counter == 0:
                for col in row:
                    header.append(col)
            else:
                rows.append([int(row[0]), float(row[1]), float(row[2])])
            counter += 1
        rows.sort(key=lambda x: x[0])
        return header,rows

def sort_csv_file(file_name):
    header, rows = sort_csv_data(file_name)
    with open(file_name, 'w', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerow(header) # 4. write the header
        csvwriter.writerows(rows) # 5. write the rest of the data


def validate_agent(env, agent_path, num_of_val_plays, saveName):
    allCheckpoints = [f for f in listdir(agent_path) if isfile(join(agent_path, f))]
    filename = "agent_validations/" + saveName + ".csv"
    data = []
    for checkpoint in allCheckpoints:
        model = PPO2.load(agent_path +checkpoint)
        step = int(checkpoint.replace("mario_","").replace(".zip",""))
        returnScore, winScore = play(num_of_val_plays, env, model)
        data.append([step, returnScore, winScore/num_of_val_plays])
    header = ['Steps', 'Avg. Return', 'WinRate']
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerow(header) # 4. write the header
        csvwriter.writerows(data) # 5. write the rest of the data
    sort_csv_file(filename)


def run_arl(arl: ARLPCG, generate_num, try_num, aux):
    arl.set_aux(aux)
    wins = 0
    return_score = 0
    avg_length = 0
    num_of_fails = 0
    success = False
    for i in range(generate_num):
        arl.level = arl.generate_level(True)
        avg_length += len(arl.level)
        levelString = arl.util_convert_level_to_string()
        for env in arl.env_solver.envs:
            success = env.setLevel(levelString)
            env.setARLLevel(arl.level)
        if not success:
            num_of_fails += 1
        if success:
            for j in range(try_num):
                done = [False]
                obs = arl.env_solver.reset()
                while not done[0]:
                    action, _states = arl.solver.predict(obs)
                    obs, rewards, done, info = arl.env_solver.step(action)
                    #arl.env_solver.render()
                    if done[0]:
                        return_score += float(info[0]["ReturnScore"])
                        if info[0]["Result"] == "Win":
                            wins += 1
    avg_length = avg_length/generate_num
    wins = wins/(generate_num*try_num)
    return_score = return_score/(generate_num*try_num)
    return [aux, wins, return_score, avg_length, num_of_fails/generate_num]



def validate_arl(arl: ARLPCG, generate_num, try_num, saveName):
    results = []
    results.append(run_arl(arl, generate_num, try_num, -1))
    results.append(run_arl(arl, generate_num, try_num, -0.5))
    results.append(run_arl(arl, generate_num, try_num, 0))
    results.append(run_arl(arl, generate_num, try_num, 0.5))
    results.append(run_arl(arl, generate_num, try_num, 1))

    filename = "arl_validations/" + saveName + ".csv"
    data = []
    header = ['Aux-input', 'WinRate', 'Avg. Return', "Avg. Length", "Per. Invalid Levels"]
    with open(filename, 'w', newline="") as file:
        csvwriter = csv.writer(file) # 2. create a csvwriter object
        csvwriter.writerow(header) # 4. write the header
        csvwriter.writerows(results) # 5. write the rest of the data
    #sort_csv_file(filename)

    

