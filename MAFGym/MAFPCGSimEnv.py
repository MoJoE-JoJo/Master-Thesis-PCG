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
    WINRATE_MAP = 2
    AVG_RETURN_MAP = 3
    WINRATE_MAP_FAIL = 4


class MAFPCGSimEnv(gym.Env):
    """OpenAI Gym Environment for generating levels for the Mario AI Framework"""
    metadata = {'render.modes': ['human']}
    slice_ids = []
    num_of_slices = 0
    aux_input = 0
    farthest_slice_id = 0
    reward_type = GeneratorRewardType.WINRATE_MAP_FAIL

    total_reward = 0
    min = 10
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
            if self.reward_type != GeneratorRewardType.WINRATE_MAP_FAIL:
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
            valid_level = False
            for env in self.solver_env.envs:
                valid_level = env.setLevel(level_string)
            #self.solver_env.envs[0].setARLLevel(self.slice_ids)
            returns = []
            wins = 0
            if self.run_sim:
                if self.reward_type == GeneratorRewardType.WINRATE_MAP_FAIL:
                    if valid_level == False:
                        win_rate = -1
                        print("Invalid level generated")
                    if valid_level == True:
                        avg_return, win_rate = self.run_simulation()    
                else:
                    avg_return, win_rate = self.run_simulation()
        

        reward = self.reward(action, avg_return, win_rate, done)
        self.total_reward += reward

        self.num_of_slices = len(self.slice_ids)
        self.state = [self.num_of_slices, self.min, self.max, self.aux_input, self.slice_ids[-1]]
        
        #print("Reward: " + str(reward))
        dict = {"Yolo": "Yolo", "Result" : "Result", "ReturnScore": "ReturnScore"}
        
        return self.state, reward, done, dict

    def run_simulation(self):
        avg_return = 0
        win_rate = 0
        returns = []
        wins = 0
        if(len(self.solver_env.envs) == 1):
            num_of_sim = 5
            if self.reward_type == GeneratorRewardType.WINRATE_MAP or self.reward_type == GeneratorRewardType.WINRATE_MAP_FAIL:
                num_of_sim = 10
            for i in range(num_of_sim):
                solver_obs = self.solver_env.reset()
                solver_done = False
                while not solver_done:
                    solver_action, _states = self.solver_agent.predict(solver_obs)
                    solver_obs, solver_reward, solver_done, solver_info = self.solver_env.step(solver_action)
                    solver_done = solver_done[0]
                    if solver_done:
                        #obs = self.solver_env.reset()
                        return_score = float(solver_info[0]["ReturnScore"])
                        if solver_info[0]["Result"] == "Win":
                            wins += 1
                        returns.append(return_score)
            avg_return = sum(returns)/num_of_sim
            win_rate = wins/num_of_sim
        elif(len(self.solver_env.envs) > 1):
            #Iterative should now be handled
            solver_envs_done = [False, False, False, False, False, False, False, False, False, False]
            num_of_dones = 0
            solver_obs = self.solver_env.reset()
            solver_done = False
            while num_of_dones < 10:
                solver_action, _states = self.solver_agent.predict(solver_obs)
                solver_obs, solver_reward, solver_dones, solver_info = self.solver_env.step(solver_action)
            
                for i in range(len(solver_dones)):
                    if solver_dones[i]:
                        if(solver_envs_done[i] == False):
                            num_of_dones += 1
                            return_score = float(solver_info[i]["ReturnScore"])
                            if solver_info[i]["Result"] == "Win":
                                wins += 1
                            returns.append(return_score)
                            solver_envs_done[i] = True
            
            avg_return = sum(returns)/10
            win_rate = wins/10
        return avg_return, win_rate

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
            return self.binary_rew(avg_return, done)
        elif self.reward_type == GeneratorRewardType.NORMAL:
            external_rew = self.external_factor * avg_return * self.aux_input
            dup_rew = self.dup_rew(action)
            start_rew = self.start_rew(action)
            end_rew = self.end_rew(action)
            internal_rew = self.internal_factor * (dup_rew + start_rew + end_rew)
            return external_rew + internal_rew
        elif self.reward_type == GeneratorRewardType.WINRATE_MAP:
            external_rew = self.winrate_rew(win_rate, done) * self.external_factor
            dup_rew = self.dup_rew(action)
            start_rew = self.start_rew(action)
            end_rew = self.end_rew(action)
            internal_rew = self.internal_factor * (dup_rew + start_rew + end_rew)
            return external_rew + internal_rew
        elif self.reward_type == GeneratorRewardType.WINRATE_MAP_FAIL:
            external_rew = 0
            if win_rate == -1:
                external_rew = -1000 * self.external_factor
            else:
                external_rew = self.winrate_rew(win_rate, done) * self.external_factor
            dup_rew = self.dup_rew(action)
            start_rew = self.start_rew(action)
            end_rew = self.end_rew(action)
            internal_rew = self.internal_factor * (dup_rew + start_rew + end_rew)
            return external_rew + internal_rew
        elif self.reward_type == GeneratorRewardType.AVG_RETURN_MAP:
            external_rew = self.avg_return_rew(avg_return, done) * self.external_factor
            dup_rew = self.dup_rew(action)
            start_rew = self.start_rew(action)
            end_rew = self.end_rew(action)
            internal_rew = self.internal_factor * (dup_rew + start_rew + end_rew)
            return external_rew + internal_rew

    def winrate_rew(self, win_rate, done):
        reward = 0
        if done:
            if self.aux_input == -1:
                if  win_rate < 0.05:
                    reward = 0
                elif win_rate >= 0.05 and win_rate <= 0.15:
                    reward = 2500
                elif win_rate >= 0.15 and win_rate <= 0.25:
                    reward = 2000
                elif win_rate >= 0.25 and win_rate <= 0.35:
                    reward = 1500
                elif win_rate >= 0.35 and win_rate <= 0.45:
                    reward = 1000
                elif win_rate >= 0.45 and win_rate <= 0.55:
                    reward = 500
            elif self.aux_input == -0.5:
                if  win_rate < 0.05:
                    reward = 0
                elif win_rate >= 0.05 and win_rate < 0.15:
                    reward = 1500
                elif win_rate >= 0.15 and win_rate < 0.25:
                    reward = 2000
                elif win_rate >= 0.25 and win_rate <= 0.35:
                    reward = 2500
                elif win_rate >= 0.35 and win_rate <= 0.45:
                    reward = 2000
                elif win_rate >= 0.45 and win_rate <= 0.55:
                    reward = 1500
                elif win_rate >= 0.55 and win_rate <= 0.65:
                    reward = 1000
                elif win_rate >= 0.65 and win_rate <= 0.75:
                    reward = 500
            elif self.aux_input == 0.5:
                if win_rate >= 0.25 and win_rate < 0.35:
                    reward = 500
                elif win_rate >= 0.35 and win_rate < 0.45:
                    reward = 1000
                elif win_rate >= 0.45 and win_rate < 0.55:
                    reward = 1500
                elif win_rate >= 0.55 and win_rate < 0.65:
                    reward = 2000
                elif win_rate >= 0.65 and win_rate <= 0.75:
                    reward = 2500
                elif win_rate >= 0.75 and win_rate <= 0.85:
                    reward = 2000
                elif win_rate >= 0.85 and win_rate <= 0.95:
                    reward = 1500
                elif win_rate > 0.95:
                    reward = 1000
            elif self.aux_input == 1:
                if win_rate >= 0.45 and win_rate < 0.55:
                    reward = 500
                elif win_rate >= 0.55 and win_rate < 0.65:
                    reward = 1000
                elif win_rate >= 0.65 and win_rate < 0.75:
                    reward = 1500
                elif win_rate >= 0.75 and win_rate < 0.85:
                    reward = 2000
                elif win_rate >= 0.85 and win_rate <= 0.95:
                    reward = 2500
                elif win_rate > 0.95:
                    reward = 2000
        return reward

    def avg_return_rew(self, avg_return, done):
        reward = 0
        if done:
            if self.aux_input == -1:
                if  avg_return < 100:
                    reward = 0
                elif avg_return >= 100 and avg_return <= 320:
                    reward = 2500
                elif avg_return >= 321 and avg_return <= 519:
                    reward = 2000
                elif avg_return >= 520 and avg_return <= 740:
                    reward = 1500
                elif avg_return >= 741 and avg_return <= 939:
                    reward = 1000
                elif avg_return >= 940 and avg_return <= 1160:
                    reward = 500
            elif self.aux_input == -0.5:
                if  avg_return < 100:
                    reward = 0
                elif avg_return >= 100 and avg_return <= 320:
                    reward = 1500
                elif avg_return >= 321 and avg_return <= 519:
                    reward = 2000
                elif avg_return >= 520 and avg_return <= 740:
                    reward = 2500
                elif avg_return >= 741 and avg_return <= 939:
                    reward = 2000
                elif avg_return >= 940 and avg_return <= 1160:
                    reward = 1500
                elif avg_return >= 1161 and avg_return <= 1359:
                    reward = 1000
                elif avg_return >= 1360 and avg_return <= 1580:
                    reward = 500
            elif self.aux_input == 0.5:
                if avg_return >= 520 and avg_return <= 740:
                    reward = 500
                elif avg_return >= 741 and avg_return <= 939:
                    reward = 1000
                elif avg_return >= 940 and avg_return <= 1160:
                    reward = 1500
                elif avg_return >= 1161 and avg_return <= 1359:
                    reward = 2000
                elif avg_return >= 1360 and avg_return <= 1580:
                    reward = 2500
                elif avg_return >= 1581 and avg_return <= 1779:
                    reward = 2000
                elif avg_return >= 1780 and avg_return <= 2000:
                    reward = 1500
                elif avg_return > 2000:
                    reward = 1000
            elif self.aux_input == 1:
                if avg_return >= 940 and avg_return <= 1160:
                    reward = 500
                elif avg_return >= 1161 and avg_return <= 1359:
                    reward = 1000
                elif avg_return >= 1360 and avg_return <= 1580:
                    reward = 1500
                elif avg_return >= 1581 and avg_return <= 1779:
                    reward = 2000
                elif avg_return >= 1780 and avg_return <= 2000:
                    reward = 2500
                elif avg_return > 2000:
                    reward = 2000
        return reward

    def binary_rew(self, avg_return, done):
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