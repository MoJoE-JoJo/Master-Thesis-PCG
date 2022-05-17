import os
import time
from os import listdir
from os.path import isfile, join
from ARLPCG2 import ARLPCG2, SolverType
from Validate_Agents import validate_arl2
from MAFGym.util import readLevelFile



#--------------------------------------------------------------------------------------------
generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"

arl_save_folder = "saved_arl/41/"
arl= None
arl = ARLPCG2(
    load_path="saved_arl/41/arl-dev_499.zip", 
    generate_path=generated_level_path, 
    save_name="arl-dev", 
    internal=1, 
    external=1,
    gen_steps=400,
    sol_steps=512, 
    aux_switch=10,
    solver_type=SolverType.LEARNING)

level_folder ="MAFGym/levels/original/subset/simplified/custom/"
onlyfiles = [f for f in listdir(level_folder) if isfile(join(level_folder, f))]

bad_files = 0

for f in onlyfiles:
    file = open(level_folder+f, 'r')
    lines = file.readlines()

    length = len(lines[0])
    index = 0
    old_height = 0

    for i in range(length):
        height = 0
        for j in range(16):
            char = lines[15-j][i]
            if char != 'X':
                break
            elif char == 'X':
                height += 1          
        if height > old_height + 4:
            bad_files += 1
            break
        old_height = height

    
print("Bad files: " + str(bad_files) + " : Percent Bad: " + str(bad_files/len(onlyfiles)))