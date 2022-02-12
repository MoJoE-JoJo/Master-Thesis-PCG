import subprocess
from py4j.java_gateway import JavaGateway
import threading
import keyboard
from MAFGym.util import readLevelFile
from MAFGym.MAFEnv import MAFEnv
import os

levelFilePath = os.path.dirname(os.path.realpath(__file__)) + "\\MAFGym\\levels\\original\\lvl-1.txt"
levelString = readLevelFile(levelFilePath)


gateway = JavaGateway() 
marioGym = gateway.entry_point
print(os.path.dirname(os.path.realpath(__file__)))
current_dir = os.path.dirname(os.path.realpath(__file__))
subprocess.call([current_dir + '\\MAFGym\\RunJar.bat'])

def startMarioGym():
    marioGym.playGame(levelString, 20, 0, True)


x = threading.Thread(target=startMarioGym)
x.start()

while True:
    left, right, down, speed, jump = False, False, False, False, False

    if keyboard.is_pressed("left"):
        left = True
    if keyboard.is_pressed("right"):
        right = True
    if keyboard.is_pressed("down"):
        down = True
    if keyboard.is_pressed("a"):
        speed = True
    if keyboard.is_pressed("s"):
        jump = True
    marioGym.agentInput(left, right, down, speed, jump)