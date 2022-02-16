from os import listdir
from os.path import isfile, join




def loadLevels(folder_path):
    onlyFiles = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
    slices = []
    for file in onlyFiles:
        more_slices = True
        open_file = open(folder_path+ "/" + file, 'r')
        lines = open_file.readlines()
        counter = 0
        while more_slices:
            slice = []
            for line in lines:
                string_line = ""
                if(line[counter+15] == "\n"): 
                    more_slices = False
                    break
                else:
                    for i in range(counter, counter+16):
                        string_line += line[i]
                        #more_slices = False
                    slice.append(string_line)
                #line = line[1::]
            counter+=1
            if slice != []: slices.append(slice)
            #if("\n" in slice[0]):
            #    more_slices = False
        print("Yolo")

level_folder ="MAFGym/levels/original"
loadLevels(level_folder)

def sliceComparison(slice1, slice2):
    string1 = "".join(slice1)
    string2 = "".join(slice2)
    return string1 == string2