from MAFGym.MAFPCGEnv import PCGObservationType
import Validate_Agents
import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG import ARLPCG, PCGEnvType, SolverType
from Validate_Agents import validate_arl
from MAFGym.util import readLevelFile
import random

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


arl= None
arl = ARLPCG(
    load_path="saved_arl/31/arl-dev_245.zip", 
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

validate_arl(arl, 100, 1, "31_analysis")

#for i in range(100):
#    arl.generate_level_to_file()