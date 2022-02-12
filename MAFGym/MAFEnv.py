import gym
from gym import spaces
import numpy as np
import subprocess
from py4j.java_gateway import JavaGateway
import os

import numpy
from numpy  import array

class MAFEnv(gym.Env):
  """OpenAI Gym Environment for using the Mario AI Framework"""
  metadata = {'render.modes': ['human']}
  gateway = JavaGateway() 
  marioGym = gateway.entry_point
  useRender = False

  def __init__(self, levelFile, gameTime, initRender):
    """
    Constructs the MAFEnv Gym Environment object.
    
    :param levelFile: String representing the level.
    
    :param gameTime: The amount of seconds that the game should be allowed to run for.
    
    :param initRender: Whether or not to initialize the renderer. Should be set to True if Render() is to be called, otherwise it can be set to False, and the environment will initialize a little faster.
    """
    super(MAFEnv, self).__init__()
    self.useRender = initRender
    self.action_space = spaces.MultiBinary(5)
    self.observation_space = spaces.Box(low=-100, high=100, shape=(16, 16), dtype=np.uint8)
    print(os.path.dirname(os.path.realpath(__file__)))
    current_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.call([current_dir + '\\RunJar.bat'])
    self.marioGym.init(levelFile, current_dir + "\\img\\", gameTime, 0, self.useRender)


  def step(self, action):
    # Execute one time step within the environment
    LEFT,RIGHT,DOWN,SPEED,JUMP = bool(action[0]), bool(action[1]), bool(action[2]), bool(action[3]), bool(action[4])
    returnVal = self.marioGym.step(LEFT,RIGHT,DOWN,SPEED,JUMP)
    javaState = returnVal.getState()
    state = np.frombuffer(javaState, dtype=np.int32)
    state = state.reshape((16, 16))
    javaDict = returnVal.getInfo()
    dict = {"Yolo": javaDict.get("Yolo")}
    return state, returnVal.getReward(), returnVal.getDone(), dict

  def reset(self):
    # Reset the state of the environment to an initial state
    returnVal = self.marioGym.reset(self.useRender)
    javaState = returnVal.getState()
    state = np.frombuffer(javaState, dtype=np.int32)
    state = state.reshape((16, 16))
    return state

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    self.marioGym.render()
  
  def setLevel(self, levelString):
    self.marioGym.setLevel(levelString)