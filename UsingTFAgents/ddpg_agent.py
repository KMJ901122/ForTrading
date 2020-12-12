from StockEnvironment import StockEnv
from datapreprocess import StatesPreprocess
import tensorflow_probability as tfp
from customActorNetwork import CustomActorNetwork
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network, normal_projection_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import greedy_policy, random_tf_policy, gaussian_policy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from Evaluations import compute_avg_return


actor_fc_layer_params = (32, )
critic_joint_fc_layer_params = (32, )

preprocessing_layers=tf.keras.models.Sequential(
[
tf.keras.layers.Reshape((window_size, -1)),
tf.keras.layers.Conv1D(10, 5),
tf.keras.layers.LSTM(32),
tf.keras.layers.Flatten()
])

from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network

actor_net=CustomActorNetwork(train_env.observation_spec(),
train_env.action_spec(), preprocessing_layers=preprocessing_layers,
fc_layer_params=actor_fc_layer_params)

critic_net = critic_network.CriticNetwork(
    (train_env.observation_spec(),  train_env.action_spec()),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params)

global_step = tf.compat.v2.Variable(0)

tf_agent = ddpg_agent.DdpgAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)
