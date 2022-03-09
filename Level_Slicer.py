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
    return slices

def sliceComparison(slice1, slice2):
    string1 = "".join(slice1)
    string2 = "".join(slice2)
    return string1 == string2


def cleanSlicesTubes(slices):
    new_slices = []
    for slice in slices:
        malformed_slice = False
        for line in slice:
            if (line[0] == "t" and line[1] != "t") or (line[0] == "T" and line[1] != "T") or (line[15] == "t" and line[14] != "t") or (line[15] == "T" and line[14] != "T"):
                malformed_slice = True
        if not malformed_slice:
            new_slices.append(slice)
    return new_slices

def cleanSlicesDuplicates(slices):
    new_slices = []
    for slice in slices:
        if slice not in new_slices:
            new_slices.append(slice)
    return new_slices

def makeSlices(folder_path):
    slices = loadLevels(folder_path)
    slices = cleanSlicesTubes(slices)
    slices = cleanSlicesDuplicates(slices)
    return slices

