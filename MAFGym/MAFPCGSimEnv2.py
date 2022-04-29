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

class MAFPCGSimEnv2(gym.Env):
    """OpenAI Gym Environment for generating levels for the Mario AI Framework"""
    metadata = {'render.modes': ['human']}
    actions = []
    num_of_slices = 0
    aux_input = 0

    total_reward = 0
    max_actions = 320
    observation = []

    state = []

    generate_folder_path = ""

    internal_factor = 1
    external_factor = 1
    solver_env = None
    solver_agent = None
    run_sim = True

    def __init__(self, aux, generate_path, solver_env:MAFEnv, solver_agent):
        super(MAFPCGSimEnv2, self).__init__()
        self.generate_folder_path = generate_path
        self.solver_env = solver_env
        self.solver_agent = solver_agent
        
        self.aux_input = aux
        self.action_space = spaces.Discrete(17)
        low = np.array([16,-1])
        high = np.array([320,1])
        for n in range(256):
            low = np.append(low, 0)
            high = np.append(high, 1)
        self.observation_space = spaces.Box(low=low, high=high, shape=(258,), dtype=np.int32)
        self.state = self.reset()
    

    def step(self, action):
        action = int(action)
        done = False
        if(action == 16):
            done = True
        elif(self.num_of_slices == self.max_actions):
            done = True
            action = 16
        self.actions.append(action)
        self.num_of_slices = len(self.actions)


        win_rate = 0
        avg_return = 0

        if(done):
            if self.run_sim:
                #Run simulation
                level_string = self.actions_to_string(self.actions)
                for env in self.solver_env.envs:
                    valid_level = env.setLevel(level_string)
                if not valid_level:
                    print("!!!WHAT THE FUCK, INVALID LEVEL!!!")  
                avg_return, win_rate = self.run_simulation()
    
        reward = self.reward(action, avg_return, win_rate, done)
        self.total_reward += reward

        self.observation.append(action)
        self.observation.pop(0)
        obs = []
        for act in self.observation:
            obs += self.action_to_ints(act)
        
        self.state = [self.num_of_slices, self.aux_input] + obs
        
        #print("Reward: " + str(reward))
        dict = {"Yolo": "Yolo", "Result" : "Result", "ReturnScore": "ReturnScore"}
        
        return self.state, reward, done, dict

    def run_simulation(self):
        avg_return = 0
        win_rate = 0
        returns = []
        wins = 0
        if(len(self.solver_env.envs) == 1):
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
        actions_string = "["
        for action in self.actions:
            actions_string += str(action)+", "
        actions_string += "]"
        print("Reset : Return: " + str(self.total_reward) + " : Aux: " + str(self.aux_input) + " : Slice_Ids: " + actions_string)
        self.actions = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        self.num_of_slices = 16
        self.total_reward = 0
        start_obs = [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,
        ]
        self.observation = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]
        self.state = [self.num_of_slices, self.aux_input] + start_obs
        return self.state

    def action_to_ints(self, action):
        ints = []
        if action == 0:
            ints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        elif action == 1:
            ints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
        elif action == 2:
            ints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
        elif action == 3:
            ints = [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
        elif action == 4:
            ints = [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]
        elif action == 5:
            ints = [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1]
        elif action == 6:
            ints = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1]
        elif action == 7:
            ints = [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1]
        elif action == 8:
            ints = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
        elif action == 9:
            ints = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1]
        elif action == 10:
            ints = [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1]
        elif action == 11:
            ints = [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
        elif action == 12:
            ints = [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
        elif action == 13:
            ints = [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1]
        elif action == 14:
            ints = [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        elif action == 15:
            ints = [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        #if action == 16:
        #    ints = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1]
            
        return ints

    def actions_to_string(self, actions):
        lines = [""] * 16
        for i in range(len(actions)):
            if i == 0:
                lines[0] = "-"
                lines[1] = "-"
                lines[2] = "-"
                lines[3] = "-"
                lines[4] = "-"
                lines[5] = "-"
                lines[6] = "-"
                lines[7] = "-"
                lines[8] = "-"
                lines[9] = "-"
                lines[10] = "-"
                lines[11] = "-"
                lines[12] = "-"
                lines[13] = "M"
                lines[14] = "X"
                lines[15] = "X"
            elif i == len(actions)-1:
                lines[0] += "-"
                lines[1] += "-"
                lines[2] += "-"
                lines[3] += "-"
                lines[4] += "-"
                lines[5] += "-"
                lines[6] += "-"
                lines[7] += "-"
                lines[8] += "-"
                lines[9] += "-"
                lines[10] += "-"
                lines[11] += "-"
                lines[12] += "-"
                lines[13] += "F"
                lines[14] += "X"
                lines[15] += "X"
            else:
                ints = self.action_to_ints(actions[i])
                for j in range(len(ints)):
                    if ints[j] == 0:
                        lines[j] += "-"
                    elif ints[j] == 1:
                        lines[j] += "X"
        level_string = ""
        for line in lines:
            level_string += line
            level_string += "\n"
        return level_string

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        #Write the file
        #self.slice_ids.append(1)
        #self.slice_ids.append(1)
        #self.slice_ids.append(1)
        
        #lines = [""] * 16
        #for slice_id in self.slice_ids:
        #    slice = self.slice_map.get(slice_id)
        #    for line_index in range(len(slice)):
        #        lines[line_index] += slice[line_index]
        #        lines[line_index] += "+"
             
        #if not os.path.exists(self.generate_folder_path):
        #    os.makedirs(self.generate_folder_path)

        #open_file = open(self.generate_folder_path + str(time.time_ns())+ ".txt", 'w')
        #for line in lines:
        #    open_file.write(line)
        #    open_file.write("\n")
        print("Tried to render")
        

    def reward(self, action, avg_return, win_rate, done):
        external_rew = self.winrate_rew(win_rate, done) * self.external_factor
        #external_rew += self.avg_return_rew(avg_return, done) * self.external_factor
        internal_rew = 0
        if action > self.actions[-2] + 4 and action != 16:
            internal_rew = -100 * self.internal_factor
        else:
            internal_rew = 5 * self.internal_factor
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