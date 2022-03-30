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
import enum
import json

import numpy

class PCGObservationType(enum.Enum):
    ID = 0
    GRID = 1


class MAFPCGEnv(gym.Env):
    """OpenAI Gym Environment for generating levels for the Mario AI Framework"""
    metadata = {'render.modes': ['human']}
    slice_ids = []
    num_of_slices = 0
    aux_input = 0
    farthest_slice_id = 0

    total_reward = 0
    min = 10
    max = 20

    state = []

    start_set = []
    mid_set = []
    end_set = []

    slice_map = {}
    id_map = {}
    perf_map = {}
    generate_folder_path = ""

    internal_factor = 1
    external_factor = 1
    observation_type = None

    def __init__(self, aux, start_slices, mid_slices, end_slices, slice_map, generate_path, id_map, observationType:PCGObservationType ):
        super(MAFPCGEnv, self).__init__()
        self.generate_folder_path = generate_path
        self.observation_type = observationType
        self.start_set = start_slices
        self.mid_set = mid_slices
        self.end_set = end_slices
        self.slice_map = slice_map
        self.id_map = id_map
        
        self.aux_input = aux
        self.action_space = spaces.Discrete(len(self.slice_map.keys()))
        if(self.observation_type == PCGObservationType.ID):
            low = np.array([0,0,0,-1,-1])
            high = np.array([40,20,20,1,len(self.slice_map.keys())])
            self.observation_space = spaces.Box(low=low, high=high, shape=(5,), dtype=np.int32)
        elif(self.observation_type == PCGObservationType.GRID):
            low = np.array([0,0,0,-1])
            high = np.array([40,20,20,1])

            al1 = np.zeros(16)
            al1[0] = 0
            al2 = np.zeros(16)
            al2[0] = 0
            al3 = np.zeros(16)
            al3[0] = 0
            al4 = np.zeros(16)
            al4[0] = -1
            al5 = np.full((16,16),-1)
            low = [al1,al2,al3,al4]

            ah1 = np.zeros(16)
            ah1[0] = 40
            ah2 = np.zeros(16)
            ah2[0] = 20
            ah3 = np.zeros(16)
            ah3[0] = 20
            ah4 = np.zeros(16)
            ah4[0] = 1
            ah5 = np.full((16,16),31)
            high = [ah1,ah2,ah3,ah4]
            for i in range(16):
                low = np.vstack([low, al5[i]])
                high = np.vstack([high, ah5[i]])
            self.observation_space = spaces.Box(low=low, high=high, shape=(20,16), dtype=np.int32)
        self.state = self.reset()
    

    def step(self, action):
        action = int(action)
        reward = self.reward(action)
        self.total_reward += reward
        self.slice_ids.append(action)

        self.num_of_slices = len(self.slice_ids)
        if(self.observation_type == PCGObservationType.ID):
            self.state = [self.num_of_slices, self.min, self.max, self.aux_input, self.slice_ids[-1]]
        elif(self.observation_type == PCGObservationType.GRID):
            a1 = np.zeros(16)
            a1[0] = self.num_of_slices
            a2 = np.zeros(16)
            a2[0] = self.min
            a3 = np.zeros(16)
            a3[0] = self.max
            a4 = np.zeros(16)
            a4[0] = self.aux_input
            a5 = self.util_convert_string_slice_to_integers(self.slice_map[self.slice_ids[-1]])
            self.state = [a1,a2,a3,a4]
            for a in a5:
                self.state.append(a)
        done = False

        if(self.observation_type == PCGObservationType.ID):
            if(action in self.end_set):
                done = True
            if(action in self.start_set and self.state[0] > 1):
                done = True
            if(self.state[0] > self.state[2]*2):
                done = True
        elif(self.observation_type == PCGObservationType.GRID):
            if(action in self.end_set):
                done = True
            if(action in self.start_set and self.state[0][0] > 1):
                done = True
            if(self.state[0][0] > self.state[2][0]*2):
                done = True
        #print("Reward: " + str(reward))
        dict = {"Yolo": "Yolo", "Result" : "Result", "ReturnScore": "ReturnScore"}
        if(done):
            if(self.slice_ids[0] not in self.start_set):
                print("Repaired start")
                self.slice_ids[0] = random.choice(self.start_set)
            if(self.slice_ids[-1] not in self.end_set):
                print("Repaired end")
                self.slice_ids[-1] = random.choice(self.end_set)
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
        if(self.observation_type == PCGObservationType.ID):
            self.state = [self.num_of_slices, self.min, self.max, self.aux_input, -1]
        elif(self.observation_type == PCGObservationType.GRID):
            a2 = np.zeros(16)
            a3 = np.zeros(16)
            a4 = np.zeros(16)
            a2[0] = self.min
            a3[0] = self.max
            a4[0] = self.aux_input
            self.state = [np.zeros(16), a2, a3, a4]
            a5 = np.full((16,16), -1)
            for a in a5:
                self.state.append(a)
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
        



    def reward(self, action):
        external_rew = self.external_factor * self.perf_rew(action)
        dup_rew = self.dup_rew(action)
        start_rew = self.start_rew(action)
        end_rew = self.end_rew(action)
        internal_rew = self.internal_factor * (dup_rew + start_rew + end_rew)
        return external_rew + internal_rew

    def dup_rew(self, action):
        rew = 0
        if(self.observation_type == PCGObservationType.ID):
            if action == self.state[4]: 
                rew = -10
            elif action != self.state[4]: 
                rew = 10
        elif(self.observation_type == PCGObservationType.GRID):
            if(self.state[4][0] == -1):
                pass
            else:
                slice = []
                for i in range(4,20):
                    slice.append(self.state[i].tolist())
                slice_string = json.dumps(slice)
                slice_id = self.id_map[slice_string]
                if action == slice_id: 
                    rew = -10
                elif action != slice_id: 
                    rew = 10
        return rew
        
    def start_rew(self, action):
        if(self.observation_type == PCGObservationType.ID):
            rew = 0
            if self.state[0] > 0 and action in self.start_set: 
                rew = -20 
            elif self.state[0] == 0 and action in self.start_set: 
                rew = 200
            elif self.state[0] == 0 and action not in self.start_set: 
                rew = -20
            return rew
        elif(self.observation_type == PCGObservationType.GRID):
            rew = 0
            if self.state[0][0] > 0 and action in self.start_set: 
                rew = -20 
            elif self.state[0][0] == 0 and action in self.start_set: 
                rew = 200
            elif self.state[0][0] == 0 and action not in self.start_set: 
                rew = -20
            return rew
    
    def end_rew(self, action):
        if(self.observation_type == PCGObservationType.ID):
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
        elif(self.observation_type == PCGObservationType.GRID):
            rew = 0
            if self.state[0][0] >= self.max: 
                rew  = -20
            elif self.state[0][0] < self.min-1 and action in self.end_set: 
                rew = -20
            elif self.state[0][0] == self.max-1 and action not in self.end_set: 
                rew = -20
            elif self.state[0][0] <= self.max-1 and self.state[0][0] >= self.min-1 and action in self.end_set:
                rew = 200
            return rew

    def perf_rew(self, action):
        k,m = self.perf_map[action]
        return self.aux_input * m

    def set_perf_map(self, perf_map):
        self.perf_map = perf_map
    
    def util_convert_string_slice_to_integers(self, slice):
        slice_ints = np.zeros((16,16))
        for i in range(16):
            for j in range(16):
                char = slice[i][j]
                if(char == '-'):
                    slice_ints[i][j] = 0
                if(char == 'M'):
                    slice_ints[i][j] = 1
                if(char == 'F'):
                    slice_ints[i][j] = 2
                if(char == 'y'):
                    slice_ints[i][j] = 3                
                if(char == 'Y'):
                    slice_ints[i][j] = 4
                if(char == 'E'):
                    slice_ints[i][j] = 5
                if(char == 'g'):
                    slice_ints[i][j] = 6
                if(char == 'G'):
                    slice_ints[i][j] = 7
                if(char == 'k'):
                    slice_ints[i][j] = 8       
                if(char == 'K'):
                    slice_ints[i][j] = 9
                if(char == 'r'):
                    slice_ints[i][j] = 10
                if(char == 'R'):
                    slice_ints[i][j] = 11
                if(char == 'X'):
                    slice_ints[i][j] = 12
                if(char == '#'):
                    slice_ints[i][j] = 12
                if(char == '%'):
                    slice_ints[i][j] = 13
                if(char == '|'):
                    slice_ints[i][j] = 14
                if(char == '*'):
                    slice_ints[i][j] = 15
                if(char == 'B'):
                    slice_ints[i][j] = 16
                if(char == 'b'):
                    slice_ints[i][j] = 17
                if(char == '?'):
                    slice_ints[i][j] = 18
                if(char == '@'):
                    slice_ints[i][j] = 19
                if(char == 'Q'):
                    slice_ints[i][j] = 20
                if(char == '!'):
                    slice_ints[i][j] = 21
                if(char == '1'):
                    slice_ints[i][j] = 22    
                if(char == '2'):
                    slice_ints[i][j] = 23
                if(char == 'D'):
                    slice_ints[i][j] = 24        
                if(char == 'S'):
                    slice_ints[i][j] = 25
                if(char == 'C'):
                    slice_ints[i][j] = 26     
                if(char == 'U'):
                    slice_ints[i][j] = 27
                if(char == 'L'):
                    slice_ints[i][j] = 28   
                if(char == 'o'):
                    slice_ints[i][j] = 29
                if(char == 't'):
                    slice_ints[i][j] = 30     
                if(char == 'T'):
                    slice_ints[i][j] = 31                                                                                                                                                                                                                                                                                                           
        return slice_ints