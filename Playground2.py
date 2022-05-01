from MAFGym.MAFPCGEnv import PCGObservationType
import Validate_Agents
import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG2 import ARLPCG2, SolverType
from Validate_Agents import validate_arl2
from MAFGym.util import readLevelFile

#agent_val_path = os.path.dirname(os.path.realpath(__file__)) +"\\agent_validations\\"
#all_validations = [f for f in listdir(agent_val_path) if isfile(join(agent_val_path, f))]

#for val in all_validations:
#    Validate_Agents.sort_csv_file(agent_val_path+val)

def train(arl: ARLPCG2, hours_run):
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

#--------------------------------------------------------------------------------------------
generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"

arl_save_folder = "saved_arl/41/"
arl= None
arl = ARLPCG2(
    load_path="", 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=1, 
    external=1,
    gen_steps=400,
    sol_steps=512, 
    aux_switch=10,
    solver_type=SolverType.LEARNING)

train(arl, 12)
validate_arl2(arl, 100, 10, "41_arl_12")
train(arl, 12)
validate_arl2(arl, 100, 10, "41_arl_24")
train(arl, 12)
validate_arl2(arl, 100, 10, "41_arl_36")
train(arl, 12)
validate_arl2(arl, 100, 10, "41_arl_48")
train(arl, 12)
validate_arl2(arl, 100, 10, "41_arl_60")