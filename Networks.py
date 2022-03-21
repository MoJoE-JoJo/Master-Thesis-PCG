import tensorflow as tf
from stable_baselines.common.policies import FeedForwardPolicy

layers_solver = [dict(vf=[512,512], pi=[512,512])]
class MarioSolverPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioSolverPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers_solver,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp")

layers_generator = [dict(vf=[1024,1024], pi=[1024,1024])]
class MarioGeneratorPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(MarioGeneratorPolicy, self).__init__(*args, **kwargs,
                                            net_arch=layers_generator,
                                            act_fun=tf.nn.relu,
                                            #cnn_extractor=modified_cnn,
                                            feature_extraction="mlp") 