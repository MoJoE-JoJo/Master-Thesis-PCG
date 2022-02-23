import Validate_Agents
import os
from os import listdir
from os.path import isfile, join

agent_val_path = os.path.dirname(os.path.realpath(__file__)) +"\\agent_validations\\"
all_validations = [f for f in listdir(agent_val_path+"old_disc_vec\\") if isfile(join(agent_val_path+"old_disc_vec\\", f))]

for val in all_validations:
    Validate_Agents.sort_csv_file(agent_val_path+"old_disc_vec\\"+val)