import Validate_Agents
import os
from os import listdir
from os.path import isfile, join
from ARLPCG import ARLPCG

#agent_val_path = os.path.dirname(os.path.realpath(__file__)) +"\\agent_validations\\"
#all_validations = [f for f in listdir(agent_val_path) if isfile(join(agent_val_path, f))]

#for val in all_validations:
#    Validate_Agents.sort_csv_file(agent_val_path+val)

level_folder ="MAFGym/levels/original/subset/completable/lvl-1"

generated_level_path = os.path.dirname(os.path.realpath(__file__)).replace("\\MAFGym", "") + "\\generated_levels\\"

arl = ARLPCG(load_path="", levels_path=level_folder, generate_path=generated_level_path, save_name="arl-dev")

#arl.train_generator(6400)
for i in range(1000):
    arl.train(i, 100)
arl.save("saved_arl/test/")