import enum
import Level_Slicer
import os
import pickle
import zipfile
from MAFGym.MAFPCGEnv import MAFPCGEnv
from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv
from Networks import MarioSolverPolicy, MarioGeneratorPolicy

class SolverType(enum.Enum):
    PRETRAINED = 0
    LEARNING = 1
    GAIL = 2

class ARLPCG():
    solver = None
    generator = None
    env_solver = None
    env_generator = None

    perf_map = {}
    slice_map = {}
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

    def __init__(self, load_path="", levels_path="", generate_path="", save_name = "pcg", gen_steps = 32, sol_steps = 512, solver_type = SolverType.PRETRAINED):
        self.dummyLevelString = os.path.dirname(os.path.realpath(__file__))+"\\ARLDummyLevel.txt"
        self.dummyLevelString = readLevelFile(self.dummyLevelString)
        self.solver_type = solver_type
        self.generator_steps = gen_steps
        self.solver_steps = sol_steps

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
        

    def empty_init(self, levels_path):
        slices = Level_Slicer.makeSlices(levels_path)
        self.util_make_slice_sets(slices)
        self.slice_map = self.util_make_slice_id_map(self.start_set, self.mid_set, self.end_set)
        self.util_convert_sets_to_ids(self.start_set, self.mid_set, self.end_set)
        
        self.empty_init_generator()
        self.empty_init_solver()

    def empty_init_solver(self):
        self.env_solver = self.util_make_dummyVecEnv([self.dummyLevelString])
        for env in self.env_solver.envs:
            env.set_perf_map(self.perf_map)
        
        if self.solver_type == SolverType.PRETRAINED:
            self.solver = PPO2.load(os.path.dirname(os.path.realpath(__file__))+"\\ARLStaticSolver", self.env_solver)
        elif self.solver_type == SolverType.LEARNING:
            self.solver = PPO2(MarioSolverPolicy, self.env_solver, verbose=1, n_steps=self.solver_steps, learning_rate=0.00005, gamma=0.99)  
        elif self.solver_type == SolverType.GAIL:
            raise ValueError("GAIL Solver not implemented")

    def empty_init_generator(self):
        self.env_generator = MAFPCGEnv(0,
            self.start_set, 
            self.mid_set, 
            self.end_set, 
            self.slice_map,  
            self.generate_path)
        self.env_generator.set_perf_map(self.perf_map)
        #self.perf_map[7] = 1
        self.generator = PPO2(MarioGeneratorPolicy, self.env_generator, verbose=1, n_steps=self.generator_steps, learning_rate=0.00005, gamma=0.99)

    def load(self, load_path):
        with zipfile.ZipFile(load_path) as thezip:
            with thezip.open('other.pkl',mode='r') as other_file:
                gen_env, solv_env, start, mid, end, slice_map, perf_map, train_its, save_name = pickle.load(other_file)
                self.env_generator = gen_env
                self.env_solver = solv_env
                self.start_set = start
                self.mid_set = mid
                self.end_set = end
                self.slice_map = slice_map
                self.perf_map = perf_map
                self.trained_iterations = train_its
                self.save_name = save_name
            with thezip.open('generator.zip',mode='r') as generator_file:
                self.generator = PPO2.load(generator_file, DummyVecEnv([lambda: self.env_generator]))
            with thezip.open("solver.zip", mode="r") as solver_file:
                self.solver = PPO2.load(solver_file, DummyVecEnv([lambda: self.env_solver]))



    def save(self, save_path):
        data = []
        self.generator.save(save_path+"generator")
        self.solver.save(save_path+"solver")
        data.append(self.env_generator)
        data.append(self.env_solver)
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

    def generate_level(self):
        pass

    def increment_steps_trained(self, iterations):
        self.trained_iterations += iterations

    def train(self, num_of_iterations):
        pass

    def train_generator(self, num_of_steps):
        pass

    def train_solver(self, num_of_steps):
        pass

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