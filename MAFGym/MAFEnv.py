import gym
from gym import spaces
import numpy as np
import subprocess
from py4j.java_gateway import JavaGateway
import os


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
    self.observation_space = spaces.Box(low=-100, high=100, shape=(16, 16, 1), dtype=np.uint8)
    print(os.path.dirname(os.path.realpath(__file__)))
    current_dir = os.path.dirname(os.path.realpath(__file__))
    subprocess.call([current_dir + '\\RunJar.bat'])
    self.marioGym.init(levelFile, current_dir + "\\img\\", gameTime, 0, self.useRender)


  def step(self, action):
    # Execute one time step within the environment
    LEFT,RIGHT,DOWN,SPEED,JUMP = action[0], action[1], action[2], action[3], action[4]
    returnVal = self.marioGym.step(LEFT,RIGHT,DOWN,SPEED,JUMP)
    return returnVal.getState(), returnVal.getReward(), returnVal.getDone(), returnVal.getInfo()

  def reset(self):
    # Reset the state of the environment to an initial state
    self.marioGym.reset(self.useRender)

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    self.marioGym.render()