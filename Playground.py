from MAFGym.MAFPCGEnv import PCGObservationType
import Validate_Agents
import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG import ARLPCG, PCGEnvType
from Validate_Agents import validate_arl

#agent_val_path = os.path.dirname(os.path.realpath(__file__)) +"\\agent_validations\\"
#all_validations = [f for f in listdir(agent_val_path) if isfile(join(agent_val_path, f))]

#for val in all_validations:
#    Validate_Agents.sort_csv_file(agent_val_path+val)

level_folder ="MAFGym/levels/original/subset/completable/"
arl_save_folder = "saved_arl/12/"
generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"
arl = ARLPCG(load_path="", levels_path=level_folder, generate_path=generated_level_path, save_name="arl-dev", internal=10, external=1, pcg_env_type=PCGEnvType.SIM_VEC)

total_runtime = 8*60*60
time_between_logs = 15*60
start_time = time.time()
logger_time = time.time()
arl.train(True)
arl.save(arl_save_folder)
run = True
while run:
    new_time = time.time()
    if(new_time - start_time >= total_runtime):
        arl.train(True)
        arl.save(arl_save_folder)
        run = False
    elif(new_time - logger_time >= time_between_logs):
        logger_time = new_time
        arl.train(True)
        arl.save(arl_save_folder)
    else:
        arl.train(False)
validate_arl(arl, 100, 10, "12_arl")


#--------------------------------------------------------------------------------------------
arl_save_folder = "saved_arl/13/"
generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"
arl = ARLPCG(load_path="", levels_path=level_folder, generate_path=generated_level_path, save_name="arl-dev", internal=10, external=1, pcg_env_type=PCGEnvType.SIM)

total_runtime = 8*60*60
time_between_logs = 15*60
start_time = time.time()
logger_time = time.time()
arl.train(True)
arl.save(arl_save_folder)
run = True
while run:
    new_time = time.time()
    if(new_time - start_time >= total_runtime):
        arl.train(True)
        arl.save(arl_save_folder)
        run = False
    elif(new_time - logger_time >= time_between_logs):
        logger_time = new_time
        arl.train(True)
        arl.save(arl_save_folder)
    else:
        arl.train(False)
validate_arl(arl, 100, 10, "13_arl")

