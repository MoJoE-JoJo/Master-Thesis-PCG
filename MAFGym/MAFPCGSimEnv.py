import gym
from gym import spaces
import numpy as np
import os
import random
import copy
import time
import enum
import json

import numpy
from MAFGym.MAFEnv import MAFEnv

class GeneratorRewardType(enum.Enum):
    NORMAL = 0
    BINARY = 1

class MAFPCGSimEnv(gym.Env):
    """OpenAI Gym Environment for generating levels for the Mario AI Framework"""
    metadata = {'render.modes': ['human']}
    slice_ids = []
    num_of_slices = 0
    aux_input = 0
    farthest_slice_id = 0
    reward_type = GeneratorRewardType.NORMAL

    total_reward = 0
    min = 4
    max = 20

    start_constraint = True
    end_constraint = True
    length_constraint = True
    duplication_constraint = True

    state = []

    start_set = []
    mid_set = []
    end_set = []

    slice_map = {}
    generate_folder_path = ""

    internal_factor = 1
    external_factor = 1
    solver_env = None
    solver_agent = None
    run_sim = True

    def __init__(self, aux, start_slices, mid_slices, end_slices, slice_map, generate_path, solver_env:MAFEnv, solver_agent):
        super(MAFPCGSimEnv, self).__init__()
        self.generate_folder_path = generate_path
        self.solver_env = solver_env
        self.solver_agent = solver_agent
        self.start_set = start_slices
        self.mid_set = mid_slices
        self.end_set = end_slices
        self.slice_map = slice_map
        
        self.aux_input = aux
        self.action_space = spaces.Discrete(len(self.slice_map.keys()))
        low = np.array([0,0,0,-1,-1])
        high = np.array([40,20,20,1,len(self.slice_map.keys())])
        self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.int32)
        self.state = self.reset()
    

    def step(self, action):
        action = int(action)
        self.slice_ids.append(action)
        done = False

        if(action in self.end_set):
            done = True
        if(action in self.start_set and self.state[0] >= 1):
            done = True
        if(self.state[0] >= self.state[2]*2):
            done = True
        
        avg_return = 0
        win_rate = 0

        if(done):
            if(self.slice_ids[0] not in self.start_set):
                print("Repaired start")
                self.slice_ids[0] = random.choice(self.start_set)
                self.start_constraint = False
            if(self.slice_ids[-1] not in self.end_set):
                print("Repaired end")
                self.slice_ids[-1] = random.choice(self.end_set)
                self.end_constraint = False
            #Run simulation
            level_string = self.generate_level_string()
            self.solver_env.envs[0].setLevel(level_string)
            #self.solver_env.envs[0].setARLLevel(self.slice_ids)
            returns = []
            wins = 0
            if self.run_sim:
                num_of_sim = 5
                for i in range(num_of_sim):
                    obs = self.solver_env.reset()
                    solver_done = False
                    while not solver_done:
                        action, _states = self.solver_agent.predict(obs)
                        obs, reward, solver_done, info = self.solver_env.step(action)
                        solver_done = solver_done[0]
                        if solver_done:
                            #obs = self.solver_env.reset()
                            return_score = float(info[0]["ReturnScore"])
                            if info[0]["Result"] == "Win":
                                wins += 1
                            returns.append(return_score)
                avg_return = sum(returns)/num_of_sim
                win_rate = wins/num_of_sim
        

        reward = self.reward(action, avg_return, win_rate, done)
        self.total_reward += reward

        self.num_of_slices = len(self.slice_ids)
        self.state = [self.num_of_slices, self.min, self.max, self.aux_input, self.slice_ids[-1]]
        
        #print("Reward: " + str(reward))
        dict = {"Yolo": "Yolo", "Result" : "Result", "ReturnScore": "ReturnScore"}
        
        return self.state, reward, done, dict

    def reset(self):
        # Reset the state of the environment to an initial state
        first_id = ""
        if len(self.slice_ids) > 0:
            first_id = str(self.slice_ids[0])
        slice_id_string = "["
        for id in self.slice_ids:
            slice_id_string += str(id)+", "
        slice_id_string += "]"
        print("Reset : Return: " + str(self.total_reward) + " : Aux: " + str(self.aux_input) + " : Slice_Ids: " + slice_id_string)
        self.slice_ids = []
        self.num_of_slices = 0
        self.farthest_slice_id = 0
        self.total_reward = 0
        self.start_constraint = True
        self.end_constraint = True
        self.length_constraint = True
        self.duplication_constraint = True
        self.state = [self.num_of_slices, self.min, self.max, self.aux_input, -1]
        return self.state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        #Write the file
        #self.slice_ids.append(1)
        #self.slice_ids.append(1)
        #self.slice_ids.append(1)
        lines = [""] * 16
        for slice_id in self.slice_ids:
            slice = self.slice_map.get(slice_id)
            for line_index in range(len(slice)):
                lines[line_index] += slice[line_index]
                lines[line_index] += "+"
             
        if not os.path.exists(self.generate_folder_path):
            os.makedirs(self.generate_folder_path)

        open_file = open(self.generate_folder_path + str(time.time_ns())+ ".txt", 'w')
        for line in lines:
            open_file.write(line)
            open_file.write("\n")
        



    def reward(self, action, avg_return, win_rate, done):
        if self.reward_type == GeneratorRewardType.BINARY:
            return self.binary_rew(avg_return, win_rate, done)
        elif self.reward_type == GeneratorRewardType.NORMAL:
            external_rew = self.external_factor * avg_return * self.aux_input
            dup_rew = self.dup_rew(action)
            start_rew = self.start_rew(action)
            end_rew = self.end_rew(action)
            internal_rew = self.internal_factor * (dup_rew + start_rew + end_rew)
            return external_rew + internal_rew

    def binary_rew(self, avg_return, win_rate, done):
        reward = 0
        if done:
            current_slice = -1
            for slice in self.slice_ids:
                if slice == current_slice:
                    self.duplication_constraint = False
                    break
                current_slice = slice
            if len(self.slice_ids) < self.min or len(self.slice_ids) > self.max:
                self.length_constraint = False
            
            if self.length_constraint and self.duplication_constraint and self.start_constraint and self.end_constraint:
              reward = avg_return * self.aux_input
            if not self.length_constraint:
                reward -= 2500
            if not self.duplication_constraint:
                reward -= 2500
            if not self.start_constraint:
                reward -= 2500
            if not self.end_constraint:
                reward -= 2500  
        return reward
            

    def dup_rew(self, action):
        rew = 0
        if action == self.state[4]: 
            rew = -10
        elif action != self.state[4]: 
            rew = 10
        return rew
        
    def start_rew(self, action):
        rew = 0
        if self.state[0] > 0 and action in self.start_set: 
            rew = -20 
        elif self.state[0] == 0 and action in self.start_set: 
            rew = 200
        elif self.state[0] == 0 and action not in self.start_set: 
            rew = -20
        return rew
    
    def end_rew(self, action):
        rew = 0
        if self.state[0] >= self.max: 
            rew  = -20
        elif self.state[0] < self.min-1 and action in self.end_set: 
            rew = -20
        elif self.state[0] == self.max-1 and action not in self.end_set: 
            rew = -20
        elif self.state[0] <= self.max-1 and self.state[0] >= self.min-1 and action in self.end_set:
            rew = 200
        return rew


    def set_perf_map(self, perf_map):
        pass
    
    def generate_level_string(self):
        lines = [""] * 16
        for slice_id in self.slice_ids:
            slice = self.slice_map.get(slice_id)
            for line_index in range(len(slice)):
                lines[line_index] += slice[line_index]
        level_string = ""
        for line in lines:
            level_string += line
            level_string += "\n"
        return level_string