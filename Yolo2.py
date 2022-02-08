from py4j.java_gateway import JavaGateway
import time
import os

from MAFGym.MAFEnv import MAFEnv
from MAFGym.util import readLevelFile

levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)

gymgym = MAFEnv(levelString, 100, True)
action = [False, True, False, False, False]
done = False
for i in range(100):
    timestart = time.time()
    while not done:
        obs, reward, done, info = gymgym.step(action)
    #print(obs[0][0])
        #print(reward)
        #print(done)
        #print(info[0])
        gymgym.render()
    print(time.time()-timestart)

    gymgym.reset()
    done = False


