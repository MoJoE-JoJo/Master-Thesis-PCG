from tracemalloc import start
import gym
from gym import spaces
import numpy as np
import subprocess
from py4j.java_gateway import JavaGateway
import os
import random
import copy
import time

import numpy

class MAFPCGEnv(gym.Env):
    """OpenAI Gym Environment for generating levels for the Mario AI Framework"""
    metadata = {'render.modes': ['human']}
    slice_ids = []
    num_of_slices = 0
    aux_input = 0
    farthest_slice_id = 0

    total_reward = 0
    min = 4
    max = 6

    state = []

    start_set = []
    mid_set = []
    end_set = []

    slice_map = {}
    perf_map = {}
    generate_folder_path = ""

    def __init__(self, aux, start_slices, mid_slices, end_slices, slice_map, perf_map, generate_path):
        super(MAFPCGEnv, self).__init__()
        self.generate_folder_path = generate_path
        
        self.start_set = start_slices
        self.mid_set = mid_slices
        self.end_set = end_slices
        self.slice_map = slice_map
        self.perf_map = perf_map
        
        self.aux_input = aux
        self.action_space = spaces.Discrete(len(self.slice_map.keys()))
        self.observation_space = spaces.Box(low=-1, high=len(self.slice_map.keys()), shape=(3,), dtype=np.int32)
        self.state = self.reset()
    

    def step(self, action):
        action = int(action)
        reward = self.reward(action)
        self.total_reward += reward
        self.slice_ids.append(action)

        self.num_of_slices = len(self.slice_ids)
        self.state = [self.num_of_slices, self.aux_input, self.slice_ids[-1]]

        done = False

        if(action in self.end_set):
            done = True
        if(action in self.start_set and self.state[0] > 1):
            done = True
        if(self.state[0] > self.max*10):
            done = True
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
        print("Reset : Return: " + str(self.total_reward) + " : Start: " + first_id + " : Slice_Ids: " + slice_id_string)
        self.slice_ids = []
        self.num_of_slices = 0
        self.farthest_slice_id = 0
        self.total_reward = 0
        self.state = [self.num_of_slices, self.aux_input, -1]
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
                lines[line_index] += "|"
             
        if not os.path.exists(self.generate_folder_path):
            os.makedirs(self.generate_folder_path)

        open_file = open(self.generate_folder_path + str(time.time_ns())+ ".txt", 'w')
        for line in lines:
            open_file.write(line)
            open_file.write("\n")
        



    def reward(self, action):  
        return self.dup_rew(action) + self.start_rew(action) + self.end_rew(action)

    def dup_rew(self, action):
        rew = 0
        if action == self.state[2]: 
            rew = -10
        elif action != self.state[2]: 
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




    def util_make_slice_sets(self, slices):
        for slice in slices:
            start_or_end = False
            for line in slice:
                if 'M' in line:
                    self.start_set.append(slice)
                    start_or_end = True
                    break
                if 'F' in line:
                    self.end_set.append(slice)
                    start_or_end = True
                    break
            if not start_or_end:
                self.mid_set.append(slice)
        
    def util_make_slice_id_map(self, start, mid, end):
        counter = 0
        map = {}
        for slice in start:
            map[counter] = slice
            counter += 1
        for slice in mid:
            map[counter] = slice
            counter += 1
        for slice in end:
            map[counter] = slice
            counter += 1
        return map  
    
    def util_convert_sets_to_ids(self, start, mid, end):
        for k,v in self.slice_map.items():
            if v in start:
                start.remove(v)
                start.append(k)
            elif v in mid:
                mid.remove(v)
                mid.append(k)
            elif v in end:
                end.remove(v)
                end.append(k)