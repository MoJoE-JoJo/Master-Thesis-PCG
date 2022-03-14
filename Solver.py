import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.policies import FeedForwardPolicy

layers = [dict(vf=[512,512], pi=[512,512])]

class MarioPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp")

class Solver():
    #Remember to switch to larger layers

    model = None

    def __init__(self):
        pass

    def load(self, model):
        self.model = model

    def action(self, state):
        return self.model.predict(state)