from MAFGym.MAFPCGEnv import PCGObservationType
import Validate_Agents
import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG import ARLPCG, PCGEnvType, SolverType
from Validate_Agents import validate_arl
from MAFGym.util import readLevelFile

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
ensemble = []
levelFilePath1 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-1.txt"
levelFilePath2 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-2.txt"
levelFilePath7 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-7.txt"
levelFilePath8 = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\subset\\simplified\\lvl-8.txt"
ensemble.append(readLevelFile(levelFilePath1))
ensemble.append(readLevelFile(levelFilePath2))
ensemble.append(readLevelFile(levelFilePath7))
ensemble.append(readLevelFile(levelFilePath8))


arl_save_folder = "saved_arl/37/"
arl= None
arl = ARLPCG(
    load_path="", 
    levels_path=level_folder, 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=5, 
    external=1,
    gen_steps=32, 
    aux_switch=10,
    pcg_env_type=PCGEnvType.SIM,
    solver_type=SolverType.LEARNING,
    simple_solver=True,
    ensemble=ensemble)

train(arl, 12)
validate_arl(arl, 100, 10, "36_arl_12")
train(arl, 12)
validate_arl(arl, 100, 10, "36_arl_24")
train(arl, 12)
validate_arl(arl, 100, 10, "36_arl_36")
train(arl, 12)
validate_arl(arl, 100, 10, "36_arl_48")
train(arl, 12)
validate_arl(arl, 100, 10, "36_arl_60")