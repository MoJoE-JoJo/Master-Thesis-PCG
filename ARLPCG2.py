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
from MAFGym.MAFPCGSimEnv2 import MAFPCGSimEnv2
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from Networks import MarioSolverPolicy, MarioGeneratorPolicy

class SolverType(enum.Enum):
    PRETRAINED = 0
    LEARNING = 1
    GAIL = 2

class ARLPCG2():
    solver = None
    generator = None
    env_solver = None
    env_generator = None

    generate_path = ""
    generator_steps = 0
    solver_steps = 0
    trained_iterations = 0
    save_name = ""

    dummyLevelString = ""
    solver_type = SolverType.PRETRAINED
    auxiliary = 0
    aux_values = [-1, -1, -0.5, 0.5, 1, 1]
    #aux_values = [1]
    generator_external_factor = 0
    generator_internal_factor = 0
    aux_switch_ratio = 10

    def __init__(self, load_path="", generate_path="", save_name = "pcg", gen_steps = 800, sol_steps = 512, solver_type = SolverType.PRETRAINED, external = 1, internal = 1, aux_switch = 10):
        self.dummyLevelString = os.path.dirname(os.path.realpath(__file__))+"\\ARLDummyLevel.txt"
        self.dummyLevelString = readLevelFile(self.dummyLevelString)
        self.solver_type = solver_type
        self.generator_steps = gen_steps
        self.solver_steps = sol_steps
        self.generator_internal_factor = internal
        self.generator_external_factor = external
        self.aux_switch_ratio = aux_switch

        if generate_path is "":
            self.generate_path = os.path.dirname(os.path.realpath(__file__))
        else:
            self.generate_path = generate_path

        if load_path is "":
            #do all the initialize things
            self.save_name = save_name
            self.empty_init()  
        else:
            #do all the loading things
            self.load(load_path)
        
        for env in self.env_generator.envs:
            env.internal_factor = internal
            env.external_factor = external

    def empty_init(self):
        self.empty_init_solver()
        self.empty_init_generator()

    def empty_init_solver(self):
        if(self.solver_type == SolverType.LEARNING):
            self.env_solver = self.util_make_dummyVecEnv_solver([self.dummyLevelString])
            self.solver = PPO2(MarioSolverPolicy, self.env_solver, verbose=1, n_steps=self.solver_steps, learning_rate=0.00005, gamma=0.99,tensorboard_log="logs/"+self.save_name+"-solver/")  
        
        elif(self.solver_type == SolverType.PRETRAINED):
            env1 = MAFEnv([self.dummyLevelString], 15, False)
            self.env_solver = DummyVecEnv([lambda: env1])
            self.solver = PPO2.load(os.path.dirname(os.path.realpath(__file__))+"\\SIMPLE_ARLStaticSolver", self.env_solver,tensorboard_log="logs/"+self.save_name+"-solver/")

    def empty_init_generator(self):
        self.env_generator = self.util_make_dummyVecEnv_generator_sim()
        # Non-vectorized
        # env1 = MAFPCGSimEnv2(0,
        #     self.generate_path,
        #     self.env_solver,
        #     self.solver)
        # self.env_generator = DummyVecEnv([lambda: env1])
        for env in self.env_solver.envs:
            env.perf_map = None
        #self.perf_map[7] = 1
        self.generator = PPO2(MarioGeneratorPolicy, self.env_generator, verbose=1, n_steps=self.generator_steps, learning_rate=0.00005, gamma=0.99,tensorboard_log="logs/"+self.save_name+"-generator/") 


    def load(self, load_path):
        with zipfile.ZipFile(load_path) as thezip:
            with thezip.open('other.pkl',mode='r') as other_file:
                solver_type, train_its, save_name = pickle.load(other_file)
                self.solver_type = solver_type
                self.trained_iterations = train_its
                self.save_name = save_name
            self.empty_init_solver()
            self.empty_init_generator()
            with thezip.open('generator.zip',mode='r') as generator_file:
                self.generator = PPO2.load(generator_file, self.env_generator,tensorboard_log="logs/"+self.save_name+"-generator/")
            with thezip.open("solver.zip", mode="r") as solver_file:
                self.solver = PPO2.load(solver_file, self.env_solver,tensorboard_log="logs/"+self.save_name+"-generator/")
            for env in self.env_generator.envs:
                env.solver_agent = self.solver

    def save(self, save_path):
        data = []
        self.generator.save(save_path+"generator")
        self.solver.save(save_path+"solver")
        data.append(self.solver_type)
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

    def generate_level(self, validate=False):
        obs = self.env_generator.reset()
        level = self.env_generator.envs[0].actions
        done = [False]
        if validate:
            for env in self.env_generator.envs:
                env.run_sim = False
        while not done[0]: 
            action, _states = self.generator.predict(obs)
            obs, rewards, done, info = self.env_generator.step(action)
        if validate:
            for env in self.env_generator.envs:
                env.run_sim = True
        return self.env_generator.envs[0].actions_to_string(level)

    def generate_level_to_file(self):
        level = self.generate_level(True)
                     
        if not os.path.exists(self.generate_path):
            os.makedirs(self.generate_path)

        open_file = open(self.generate_path + str(time.time_ns())+ ".txt", 'w')
        open_file.write(level)

    def increment_steps_trained(self, iterations):
        self.trained_iterations += iterations

    def set_aux(self, aux):
        self.auxiliary = aux
        for env in self.env_generator.envs:
            env.aux_input = aux

    def train(self, log_tensorboard):
        if(self.solver_type == SolverType.LEARNING):
            generator_steps = self.generator_steps * self.aux_switch_ratio * 5
            solver_steps = self.solver_steps * 100
            self.train_generator(generator_steps, log_tensorboard)
            self.train_solver(solver_steps, log_tensorboard)
            self.increment_steps_trained(1)

        elif(self.solver_type == SolverType.PRETRAINED):
            generator_steps = self.generator_steps*self.aux_switch_ratio * 5
            self.train_generator(generator_steps, log_tensorboard)
            self.increment_steps_trained(1)


    def train_generator(self, num_of_steps, log_tensorboard):
        self.env_generator.reset()
        self.env_solver.reset()
        self.auxiliary = random.choice(self.aux_values)
        for env in self.env_generator.envs:
            env.aux_input = self.auxiliary
        if (log_tensorboard):
            self.generator.tensorboard_log = "logs/"+self.save_name+"-generator/"
            self.generator.learn(num_of_steps, log_interval=100, tb_log_name="PPO-Generator", reset_num_timesteps=False)
        else:
            self.generator.tensorboard_log = None
            self.generator.learn(num_of_steps, reset_num_timesteps=False)


    def train_solver(self, num_of_steps, log_tensorboard):
        self.env_solver.reset()
        level = self.generate_level(True)
        succes = False
        for env in self.env_solver.envs:
            succes = env.setLevel(level)
        print("LEVEL VALID: " + str(succes))
        if self.solver_type is SolverType.LEARNING:
            if (log_tensorboard):
                self.solver.tensorboard_log = "logs/"+self.save_name+"-generator/"
                self.solver.learn(num_of_steps, log_interval=100, tb_log_name="PPO-Solver", reset_num_timesteps=False)
            else:
                self.solver.tensorboard_log = None
                self.solver.learn(num_of_steps, reset_num_timesteps=False)

    def util_make_dummyVecEnv_solver(self, levelStrings):
        env1 = MAFEnv(levelStrings, 30, False)
        env2 = MAFEnv(levelStrings, 30, False)  
        env3 = MAFEnv(levelStrings, 30, False)
        env4 = MAFEnv(levelStrings, 30, False)
        env5 = MAFEnv(levelStrings, 30, False)
        env6 = MAFEnv(levelStrings, 30, False)
        env7 = MAFEnv(levelStrings, 30, False)
        env8 = MAFEnv(levelStrings, 30, False)
        env9 = MAFEnv(levelStrings, 30, False)
        env10 = MAFEnv(levelStrings, 30, False)

        env_1 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10,])
        return env_1

    def util_make_dummyVecEnv_generator_sim(self):
        env1 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        env2 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        env3 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        env4 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        env5 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        #env6 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        #env7 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        #env8 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        #env9 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)
        #env10 = MAFPCGSimEnv2(0, self.generate_path, self.env_solver, self.solver)        
        env_1 = DummyVecEnv([lambda: env1,lambda: env2,lambda: env3,lambda: env4,lambda: env5])#,lambda: env6,lambda: env7,lambda: env8,lambda: env9,lambda: env10])
        return env_1