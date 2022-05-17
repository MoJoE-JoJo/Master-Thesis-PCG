from MAFGym.MAFPCGEnv import PCGObservationType
import Validate_Agents
import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG2 import ARLPCG2, SolverType
from Validate_Agents import validate_arl
from MAFGym.util import readLevelFile
import random

level_folder ="MAFGym/levels/original/subset/simplified/completable/"
#--------------------------------------------------------------------------------------------
generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\analysis\\not_trained\\"

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

for i in range(100):
    arl.generate_level_to_file()