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

max = 20
min = 10

for i in range(1001):
    length = random.randint(min,max)
    counter = 0
    level = []
    for j in range(length):
        if counter==0:
            level.append(random.choice(arl.start_set))
        elif counter== length-1:
            level.append(random.choice(arl.end_set))
        else:
            level.append(random.choice(arl.mid_set))
        counter += 1

    lines = [""] * 16
    for slice_id in level:
        slice = arl.slice_map.get(slice_id)
        for line_index in range(len(slice)):
            lines[line_index] += slice[line_index]
            
    if not os.path.exists(arl.generate_path):
        os.makedirs(arl.generate_path)

    open_file = open(arl.generate_path + "random_" + str(i) + ".txt", 'w')
    for line in lines:
        open_file.write(line)
        open_file.write("\n")