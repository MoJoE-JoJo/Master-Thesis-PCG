import Level_Slicer
import os
import tensorflow as tf
import pickle
import zipfile
from MAFGym.MAFPCGEnv import MAFPCGEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv

layers = [dict(vf=[1024,1024], pi=[1024,1024])]
class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp")  

class Generator():
    start_set = []
    mid_set = []
    end_set = []

    env = None
    model = None

    generate_path = ""
    steps_trained = 0
    save_name = ""

    slice_map = {}
    perf_map = {}


    def __init__(self, load_path="", levels_path="", generate_path="", save_name = "pcg", steps = 32, learn = 0.00005,  gamma = 0.99):
        if generate_path is "":
            self.generate_path = os.path.dirname(os.path.realpath(__file__))
        else:
            self.generate_path = generate_path

        if load_path is "":
            #do all the initialize things
            self.save_name = save_name
            slices = Level_Slicer.makeSlices(levels_path)
            self.util_make_slice_sets(slices)
            self.slice_map = self.util_make_slice_id_map(self.start_set, self.mid_set, self.end_set)
            self.util_convert_sets_to_ids(self.start_set, self.mid_set, self.end_set)
            self.env = MAFPCGEnv(0,
                self.start_set, 
                self.mid_set, 
                self.end_set, 
                self.slice_map, 
                self.perf_map, 
                self.generate_path)
            self.model = PPO2(MarioPolicy, self.env, verbose=1, n_steps=steps, learning_rate=learn, gamma=gamma)         
        else:
            #do all the loading things
            self.load(load_path)
            
    def load(self, load_path):
        with zipfile.ZipFile(load_path) as thezip:
            with thezip.open('other.pkl',mode='r') as other_file:
                env, start, mid, end, slice_map, perf_map, steps_trained, save_name = pickle.load(other_file)
                self.env = env
                self.start_set = start
                self.mid_set = mid
                self.end_set = end
                self.slice_map = slice_map
                self.perf_map = perf_map
                self.steps_trained = steps_trained
                self.save_name = save_name
            with thezip.open('model.zip',mode='r') as model_file:
                self.model = PPO2.load(model_file, DummyVecEnv([lambda: self.env]))

    def save(self, save_path):
        data = []
        self.model.save(save_path+"model")
        data.append(self.env)
        data.append(self.start_set)
        data.append(self.mid_set)
        data.append(self.end_set)
        data.append(self.slice_map)
        data.append(self.perf_map)
        data.append(self.steps_trained)
        data.append(self.save_name)
        output_file = open(save_path+"other.pkl", "wb")
        pickle.dump(data, output_file)
        output_file.close()
        zip_file = zipfile.ZipFile(save_path+self.save_name+"_"+str(self.steps_trained)+".zip", "w")
        zip_file.write(save_path+"model.zip", "model.zip")
        zip_file.write(save_path+"other.pkl","other.pkl")
        zip_file.close()
        os.remove(save_path+"model.zip")
        os.remove(save_path+"other.pkl")

    def action(self, state):
        return self.model.predict(state)

    def increment_steps_trained(self, steps):
        self.steps_trained += steps

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







    #data = []
    #data.append(train_samples)
    #data.append(train_labels)
    #data.append(test_samples)
    #data.append(test_labels)
    #with open(os.getcwd()+"\\DTModel\\dt_format_s1_f8_gsr.pkl", "wb") as output_file:
    #   pickle.dump(data, output_file)

    #with open(os.getcwd()+"\\DTModel\\dt_format_s5_f8_ann.pkl", "rb") as input_file:
    #    train_samples, train_labels, test_samples, test_labels = pickle.load(input_file)