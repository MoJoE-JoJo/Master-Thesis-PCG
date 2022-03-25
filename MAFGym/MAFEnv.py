import gym
from gym import spaces
import numpy as np
import subprocess
from py4j.java_gateway import JavaGateway
import os
import random
import copy

import numpy
from numpy  import array

class MAFEnv(gym.Env):
  """OpenAI Gym Environment for using the Mario AI Framework"""
  metadata = {'render.modes': ['human']}
  marioGym = None
  useRender = False
  arl_level = []
  levelStrings = []
  currentLevelString = ""
  gymID = 0
  perf_map = None
  death_perf_penalty = 15

  def __init__(self, levelFiles, gameTime, initRender, rewardFunction=10):
    """
    Constructs the MAFEnv Gym Environment object.
    
    :param levelFiles: List of strings representing a collection of levels.
    
    :param gameTime: The amount of seconds that the game should be allowed to run for.
    
    :param initRender: Whether or not to initialize the renderer. Should be set to True if Render() is to be called, otherwise it can be set to False, and the environment will initialize a little faster.
    
    :param sceneDetail: How much detail should the observations return regarding the blocks in the level: 0 is all detail, 1 is less detail (some different functional groupings), and 2 is binary (whether it can be jumped though or not)

    :param enemyDetail: How much detail should the observations return regarding enemies in the level: 0 is all detail, 1 is less detail (different pickup items and whether enemy can be stomped or not), and 2 is binary (friendly or enemy)
    """
    super(MAFEnv, self).__init__()
    tempID = MAFEnv.gymID
    self.gymID = tempID
    MAFEnv.gymID += 1

    print(str(self.gymID))
    self.levelStrings = levelFiles
    self.currentLevelString = random.choice(self.levelStrings)

    self.useRender = initRender
    self.action_space = spaces.Discrete(32)
    self.observation_space = spaces.Box(low=-100, high=100, shape=(16, 16, 1), dtype=np.uint8)
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if (self.gymID == 0):
      gateway = JavaGateway() 
      MAFEnv.marioGym = gateway.entry_point
      #print(os.path.dirname(os.path.realpath(__file__)))
      subprocess.call([current_dir + '\\RunJar.bat'])
    
    self.marioGym.initGym(self.gymID, self.currentLevelString, current_dir + "\\img\\", gameTime, 0, self.useRender, rewardFunction)

  def init_java_gym(self, gameTime = 60, rewardFunction = 10):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    if (self.gymID == 0):
      gateway = JavaGateway() 
      MAFEnv.marioGym = gateway.entry_point
      #print(os.path.dirname(os.path.realpath(__file__)))
      subprocess.call([current_dir + '\\RunJar.bat'])
    self.marioGym.initGym(self.gymID, self.currentLevelString, current_dir + "\\img\\", gameTime, 0, self.useRender, rewardFunction)


  def step(self, action):
    # Execute one time step within the environment
    #LEFT,RIGHT,DOWN,SPEED,JUMP = bool(action[0]), bool(action[1]), bool(action[2]), bool(action[3]), bool(action[4])
    #print(self.gymID)
    returnVal = self.marioGym.step(self.gymID, int(action))
    javaState = returnVal.getState()
    state = np.frombuffer(javaState, dtype=np.int32)
    state = state.reshape((16, 16,1))
    #state = state.reshape((4, 16, 16))
    #state = np.moveaxis(state, 0, 2)
    javaDict = returnVal.getInfo()
    dict = {"Yolo": javaDict.get("Yolo"), "Result" : javaDict.get("Result"), "ReturnScore": javaDict.get("ReturnScore")}
    done = returnVal.getDone()
    reward = returnVal.getReward()
    if(self.perf_map is not None):
      if(done):
        self.update_perf_map(returnVal.getMarioPosition(), dict)
    return state, reward, done, dict

  def update_perf_map(self, pos, dict):
    slice_done_index = pos
    slice_done_index = slice_done_index / 16
    slice_done_index = int(slice_done_index / 16)
    if(dict["Result"] == "Win"):
      k, m = self.perf_map[self.arl_level[slice_done_index]]
      k = k+1
      x = float(dict["ReturnScore"])/len(self.arl_level) 
      m = self.calcIterativeAverage(k, m, x)
      self.perf_map[self.arl_level[slice_done_index]] = (k,m)
    elif(dict["Result"] == "Lose"):
      k, m = self.perf_map[self.arl_level[slice_done_index]]
      k = k+1
      x = -1 * self.death_perf_penalty
      m = self.calcIterativeAverage(k, m, x)
      self.perf_map[self.arl_level[slice_done_index]] = (k,m)
    
    if slice_done_index is not 0:
      for index in range(slice_done_index):
        k, m = self.perf_map[self.arl_level[index]]
        k = k+1
        x = float(dict["ReturnScore"])/len(self.arl_level) 
        m = self.calcIterativeAverage(k, m, x)
        self.perf_map[self.arl_level[index]] = (k,m)

  def reset(self):
    # Reset the state of the environment to an initial state
    self.currentLevelString = random.choice(self.levelStrings)
    self.marioGym.setLevel(self.gymID, self.currentLevelString)
    returnVal = self.marioGym.reset(self.gymID, self.useRender)
    javaState = returnVal.getState()
    state = np.frombuffer(javaState, dtype=np.int32)
    state = state.reshape((16, 16,1))
    #state = state.reshape((4, 16, 16))
    #state = np.moveaxis(state, 0, 2)
    return state

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    self.marioGym.render(self.gymID)
  
  def setLevel(self, level):
    self.levelStrings = []
    self.levelStrings.append(level)
    self.marioGym.setLevel(self.gymID, level)
  
  def setARLLevel(self, level):
    self.arl_level = level

  def set_perf_map(self, perf_map):
    self.perf_map = perf_map
  
  def calcIterativeAverage(self, k, m, x):
    if(k == 1):
      return x
    else:
      return m + (x-m)/k
