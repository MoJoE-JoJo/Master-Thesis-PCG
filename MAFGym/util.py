

def readLevelFile(filepath):
    file = open(filepath, 'r')
    lines = file.readlines()
    
    levelString = ""

    for line in lines:
        levelString += line
    
    return levelString