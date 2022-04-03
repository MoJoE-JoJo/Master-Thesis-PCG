import enum
import Level_Slicer
import os
import pickle
import zipfile
import random
import time
import copy
import json
import numpy as np
from MAFGym.MAFPCGEnv import MAFPCGEnv, PCGObservationType
from MAFGym.MAFPCGSimEnv import MAFPCGSimEnv
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from Networks import MarioSolverPolicy, MarioGeneratorPolicy

class SolverType(enum.Enum):
    PRETRAINED = 0
    LEARNING = 1
    GAIL = 2

class PCGEnvType(enum.Enum):
    ID = 0
    GRID = 1
    SIM = 2

class ARLPCG():
    solver = None
    generator = None
    env_solver = None
    env_generator = None

    perf_map = {}
    slice_map = {}
    id_map = {}
    start_set = []
    mid_set = []
    end_set = []

    generate_path = ""
    generator_steps = 0
    solver_steps = 0
    trained_iterations = 0
    save_name = ""

    level = [3, 7, 174]
    dummyLevelString = ""
    solver_type = SolverType.PRETRAINED
    pcg_obs_type = PCGObservationType.ID
    pcg_env_type = PCGEnvType.ID
    auxiliary = 0
    aux_values = [-1, -1, -0.5, 0.5, 1, 1]
    generator_external_factor = 0
    generator_internal_factor = 0

    def __init__(self, load_path="", levels_path="", generate_path="", save_name = "pcg", gen_steps = 32, sol_steps = 512, solver_type = SolverType.PRETRAINED, external = 1, internal = 1, pcg_env_type = PCGEnvType.ID):
        self.dummyLevelString = os.path.dirname(os.path.realpath(__file__))+"\\ARLDummyLevel.txt"
        self.dummyLevelString = readLevelFile(self.dummyLevelString)
        self.solver_type = solver_type
        self.generator_steps = gen_steps
        self.solver_steps = sol_steps
        self.generator_internal_factor = internal
        self.generator_external_factor = external
        self.pcg_env_type = pcg_env_type
        if(pcg_env_type == PCGEnvType.GRID):
            self.pcg_obs_type = PCGObservationType.GRID
        if(pcg_env_type == PCGEnvType.ID):
            self.pcg_obs_type = PCGObservationType.ID

        if generate_path is "":
            self.generate_path = os.path.dirname(os.path.realpath(__file__))
        else:
            self.generate_path = generate_path

        if load_path is "":
            #do all the initialize things
            self.save_name = save_name
            self.empty_init(levels_path)  
        else:
            #do all the loading things
            self.load(load_path)
        
        self.env_generator.envs[0].internal_factor = internal
        self.env_generator.envs[0].external_factor = external

    def empty_init(self, levels_path):
        slices = Level_Slicer.makeSlices(levels_path)
        self.util_make_slice_sets(slices)
        self.slice_map, self.id_map = self.util_make_slice_id_map(self.start_set, self.mid_set, self.end_set)
        self.util_convert_sets_to_ids(self.start_set, self.mid_set, self.end_set)
        
        for key in self.slice_map.keys():
            self.perf_map[key] = (0,0) 
        #self.util_convert_string_slice_to_integers(self.slice_map[0])

        self.empty_init_solver()
        self.empty_init_generator()

    def empty_init_solver(self):
        if(self.solver_type == SolverType.LEARNING):
            self.env_solver = self.util_make_dummyVecEnv([self.dummyLevelString])
        elif(self.solver_type == SolverType.PRETRAINED):
            env1 = MAFEnv([self.dummyLevelString], 30, False)
            self.env_solver = DummyVecEnv([lambda: env1])
        
        for env in self.env_solver.envs:
            env.set_perf_map(self.perf_map)
            env.setARLLevel(self.level)
        
        if self.solver_type == SolverType.PRETRAINED:
            self.solver = PPO2.load(os.path.dirname(os.path.realpath(__file__))+"\\ARLStaticSolver", self.env_solver,tensorboard_log="logs/"+self.save_name+"-solver/")
        elif self.solver_type == SolverType.LEARNING:
            self.solver = PPO2(MarioSolverPolicy, self.env_solver, verbose=1, n_steps=self.solver_steps, learning_rate=0.00005, gamma=0.99,tensorboard_log="logs/"+self.save_name+"-solver/")  
        elif self.solver_type == SolverType.GAIL:
            raise ValueError("GAIL Solver not implemented")

    def empty_init_generator(self):
        if(self.pcg_env_type == PCGEnvType.GRID or self.pcg_env_type == PCGEnvType.ID):
            env1 = MAFPCGEnv(0,
                self.start_set, 
                self.mid_set, 
                self.end_set, 
                self.slice_map,  
                self.generate_path,
                self.id_map, 
                self.pcg_obs_type)
            self.env_generator = DummyVecEnv([lambda: env1])
            self.env_generator.envs[0].set_perf_map(self.perf_map)
            #self.perf_map[7] = 1
            self.generator = PPO2(MarioGeneratorPolicy, self.env_generator, verbose=1, n_steps=self.generator_steps, learning_rate=0.00005, gamma=0.99,tensorboard_log="logs/"+self.save_name+"-generator/")
        if(self.pcg_env_type == PCGEnvType.SIM):
            env1 = MAFPCGSimEnv(0,
                self.start_set, 
                self.mid_set, 
                self.end_set, 
                self.slice_map,  
                self.generate_path,
                self.env_solver,
                self.solver)
            self.env_generator = DummyVecEnv([lambda: env1])
            self.env_solver.envs[0].perf_map = None
            #self.perf_map[7] = 1
            self.generator = PPO2(MarioGeneratorPolicy, self.env_generator, verbose=1, n_steps=self.generator_steps, learning_rate=0.00005, gamma=0.99,tensorboard_log="logs/"+self.save_name+"-generator/")

    def load(self, load_path):
        with zipfile.ZipFile(load_path) as thezip:
            with thezip.open('other.pkl',mode='r') as other_file:
                solver_type, pcg_env_type, pcg_obs_type, start, mid, end, slice_map, perf_map, train_its, save_name = pickle.load(other_file)
                self.solver_type = solver_type
                self.pcg_env_type = pcg_env_type
                self.pcg_obs_type = pcg_obs_type
                self.start_set = start
                self.mid_set = mid
                self.end_set = end
                self.slice_map = slice_map
                self.perf_map = perf_map
                self.trained_iterations = train_its
                self.save_name = save_name
                #for env in self.env_solver.envs:
                #    env.init_java_gym()
            self.empty_init_solver()
            self.empty_init_generator()
            with thezip.open('generator.zip',mode='r') as generator_file:
                self.generator = PPO2.load(generator_file, self.env_generator,tensorboard_log="logs/"+self.save_name+"-generator/")
            with thezip.open("solver.zip", mode="r") as solver_file:
                self.solver = PPO2.load(solver_file, self.env_solver,tensorboard_log="logs/"+self.save_name+"-solver/")
            if(self.pcg_env_type == PCGEnvType.SIM):
                self.env_generator.envs[0].solver_agent = self.solver

    def save(self, save_path):
        data = []
        self.generator.save(save_path+"generator")
        self.solver.save(save_path+"solver")
        data.append(self.solver_type)
        data.append(self.pcg_env_type)
        data.append(self.pcg_obs_type)
        data.append(self.start_set)
        data.append(self.mid_set)
        data.append(self.end_set)
        data.append(self.slice_map)
        data.append(self.perf_map)
        data.append(self.trained_iterations)
        data.append(self.save_name)
        output_file = open(save_path+"other.pkl", "wb")
        pickle.dump(data, output_file)
        output_file.close()
        zip_file = zipfile.ZipFile(save_path+self.save_name+"_"+str(self.trained_iterations)+".zip", "w")
        zip_file.write(save_path+"generator.zip", "generator.zip")
        zip_file.write(save_path+"solver.zip", "solver.zip")
        zip_file.write(save_path+"other.pkl","other.pkl")
        zip_file.close()
        os.remove(save_path+"generator.zip")
        os.remove(save_path+"solver.zip")
        os.remove(save_path+"other.pkl")

    def action_generator(self, state):
        return self.generator.predict(state)
    
    def action_solver(self, state):
        return self.solver.predict(state)

    def generate_level(self, validate=False):
        obs = self.env_generator.reset()
        level = self.env_generator.envs[0].slice_ids
        done = [False]
        if validate:
            self.env_generator.envs[0].run_sim = False
        while not done[0]: #I think that the array shenanigans here have made perf_map better, as when generating the levels now, it can actually see when the level is properly done generating, and thus that would enable actually learning things 
            action, _states = self.generator.predict(obs)
            obs, rewards, done, info = self.env_generator.step(action)
        #level = [1, 73, 98, 102, 39, 54, 12, 90, 122, 174] #debugging magic
        if validate:
            self.env_generator.envs[0].run_sim = True
        return level

    def generate_level_to_file(self):
        level = self.generate_level()
        lines = [""] * 16
        for slice_id in level:
            slice = self.slice_map.get(slice_id)
            for line_index in range(len(slice)):
                lines[line_index] += slice[line_index]
                lines[line_index] += "+"
             
        if not os.path.exists(self.generate_path):
            os.makedirs(self.generate_path)

        open_file = open(self.generate_path + str(time.time_ns())+ ".txt", 'w')
        for line in lines:
            open_file.write(line)
            open_file.write("\n")

    def increment_steps_trained(self, iterations):
        self.trained_iterations += iterations

    def set_aux(self, aux):
        self.auxiliary = aux
        for env in self.env_generator.envs:
            env.aux_input = aux

    def train(self, log_tensorboard):
        if(self.pcg_env_type == PCGEnvType.GRID or self.pcg_env_type == PCGEnvType.ID):
            generator_steps = 32
            solver_steps = 512
            self.train_generator(generator_steps, log_tensorboard)
            self.train_solver(solver_steps)
            self.increment_steps_trained(1)
        elif(self.pcg_env_type == PCGEnvType.SIM):
            generator_steps = 32*10
            self.train_generator(generator_steps, log_tensorboard)
            self.increment_steps_trained(1)

    def train_generator(self, num_of_steps, log_tensorboard):
        self.auxiliary = random.choice(self.aux_values)
        self.env_generator.envs[0].aux_input = self.auxiliary
        obs = self.env_generator.reset()
        if (log_tensorboard):
            self.generator.tensorboard_log = "logs/"+self.save_name+"-generator/"
            self.generator.learn(num_of_steps, log_interval=100, tb_log_name="PPO-Generator", reset_num_timesteps=False)
        else:
            self.generator.tensorboard_log = None
            self.generator.learn(num_of_steps, reset_num_timesteps=False)
        self.level = self.generate_level()

    def train_solver(self, num_of_steps):
        levelString = self.util_convert_level_to_string()
        for env in self.env_solver.envs:
            env.setLevel(levelString)
            env.setARLLevel(self.level)
        if self.solver_type is SolverType.LEARNING:
            self.solver.learn(num_of_steps)
        elif self.solver_type is SolverType.PRETRAINED:
            obs = self.env_solver.reset()
            for i in range(num_of_steps):
                action, _states = self.solver.predict(obs)
                obs, rewards, done, info = self.env_solver.step(action)
                if done:
                    obs = self.env_solver.reset()

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
        slice_map = {}
        id_map = {}
        for slice in start:
            slice_map[counter] = slice
            counter += 1
        for slice in mid:
            slice_map[counter] = slice
            counter += 1
        for slice in end:
            slice_map[counter] = slice
            counter += 1
        for k,v in slice_map.items():
            int_v = self.util_convert_string_slice_to_integers(v).tolist()
            id_map[json.dumps(int_v)] = k
        return slice_map, id_map
    
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

    def util_make_dummyVecEnv(self, levelStrings):
        env1 = MAFEnv(levelStrings, 60, False)
        env2 = MAFEnv(levelStrings, 60, False)  
        env3 = MAFEnv(levelStrings, 60, False)
        env4 = MAFEnv(levelStrings, 60, False)
        env5 = MAFEnv(levelStrings, 60, False)
        env6 = MAFEnv(levelStrings, 60, False)
        env7 = MAFEnv(levelStrings, 60, False)
        env8 = MAFEnv(levelStrings, 60, False)
        env9 = MAFEnv(levelStrings, 60, False)
        env10 = MAFEnv(levelStrings, 60, False)

        env_1 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])
        return env_1

    def util_convert_level_to_string(self):
        return_string = ""
        lines = [""] * 16
        for slice_id in self.level:
            slice = self.slice_map.get(slice_id)
            for line_index in range(len(slice)):
                lines[line_index] += slice[line_index]
        
        for line in lines:
            return_string += line
            return_string += "\n"
        return return_string

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