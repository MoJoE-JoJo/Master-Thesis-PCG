import time
import os

from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile

levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)

gymgym = MAFEnv(levelString, 100, True, 1, 1)
action = [False, True, False, False, False]
done = False
for i in range(100):
    timestart = time.time()
    while not done:
        obs, reward, done, info = gymgym.step(action)
        #print(obs[0][0])
        #print(reward)
        #print(done)
        #print(info.get("Yolo"))
        gymgym.render()
    print(time.time()-timestart)

    gymgym.reset()
    done = False


