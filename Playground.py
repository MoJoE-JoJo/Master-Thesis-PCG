from MAFGym.MAFPCGEnv import PCGObservationType
import Validate_Agents
import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG import ARLPCG, PCGEnvType, SolverType
from Validate_Agents import validate_arl

#agent_val_path = os.path.dirname(os.path.realpath(__file__)) +"\\agent_validations\\"
#all_validations = [f for f in listdir(agent_val_path) if isfile(join(agent_val_path, f))]

#for val in all_validations:
#    Validate_Agents.sort_csv_file(agent_val_path+val)

def train(arl: ARLPCG, hours_run):
    total_runtime = hours_run*60*60
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

level_folder ="MAFGym/levels/original/subset/simplified/completable/"
#--------------------------------------------------------------------------------------------
generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"

arl_save_folder = "saved_arl/34/"
arl= None
arl_32_6 = ARLPCG(
    load_path="saved_arl/32/arl-dev_388.zip", 
    levels_path=level_folder, 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=5, 
    external=1,
    gen_steps=32, 
    aux_switch=10,
    pcg_env_type=PCGEnvType.SIM,
    simple_solver=True)
    #solver_type=SolverType.LEARNING)

arl_32_12 = ARLPCG(
    load_path="saved_arl/32/arl-dev_497.zip", 
    levels_path=level_folder, 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=5, 
    external=1,
    gen_steps=32, 
    aux_switch=10,
    pcg_env_type=PCGEnvType.SIM,
    simple_solver=True)
    #solver_type=SolverType.LEARNING)

arl_33_6 = ARLPCG(
    load_path="saved_arl/33/arl-dev_17512.zip", 
    levels_path=level_folder, 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=0, 
    external=1,
    gen_steps=32, 
    aux_switch=10,
    pcg_env_type=PCGEnvType.SIM,
    simple_solver=True)
    #solver_type=SolverType.LEARNING)

arl_33_12 = ARLPCG(
    load_path="saved_arl/33/arl-dev_35010.zip", 
    levels_path=level_folder, 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=0, 
    external=1,
    gen_steps=32, 
    aux_switch=10,
    pcg_env_type=PCGEnvType.SIM,
    simple_solver=True)
    #solver_type=SolverType.LEARNING)

validate_arl(arl_32_6, 100, 10, "32_2_arl_6")
validate_arl(arl_32_12, 100, 10, "32_2_arl_12")
validate_arl(arl_33_6, 100, 10, "33_2_arl_6")
validate_arl(arl_33_12, 100, 10, "33_2_arl_12")

