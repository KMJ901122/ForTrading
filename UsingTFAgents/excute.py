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

tf.compat.v1.enable_v2_behavior()

# Hyperparmeters
window_size=32
batch_size = 16
delay=True
critic_learning_rate = 3e-4
actor_learning_rate = 3e-4
alpha_learning_rate = 3e-4
target_update_tau = 0.05
policy_learning_rate = 3e-4
target_update_period = 1
gamma = 0.99
reward_scale_factor = 1.0
gradient_clipping = True
num_iterations=1000
ppo_num_iterations=200

# loss_fn=tf.math.squared_difference
# loss_fn=tf.compat.v1.losses.mean_squared_error
# loss_fn=tf.keras.losses.Huber
# loss_fn=tf.keras.losses.KLDivergence

loss_fn=tf.keras.losses.mean_absolute_error
actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate)
critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate)

actor_fc_layer_params = (64, )
critic_joint_fc_layer_params = (64, )

preprocessing_layers=tf.keras.models.Sequential(
[
tf.keras.layers.Reshape((window_size, -1)),
tf.keras.layers.Conv1D(16, 10),
tf.keras.layers.LSTM(32),
tf.keras.layers.Flatten()
])

datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'

train_states, eval_states=StatesPreprocess(datadir, window=window_size, reward_length=1)

train_py_env=StockEnv(train_states, discrete=False, delay=delay)
eval_py_env=StockEnv(eval_states, discrete=False, delay=delay, eval=True)

train_env=tf_py_environment.TFPyEnvironment(train_py_env)
eval_env=tf_py_environment.TFPyEnvironment(eval_py_env)

from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network
from TrainAndEvaluate import train_and_evaluate_ACagent

actor_net=CustomActorNetwork(train_env.observation_spec(),
train_env.action_spec(), preprocessing_layers=preprocessing_layers,
fc_layer_params=actor_fc_layer_params)

# actor_net = actor_distribution_network.ActorDistributionNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     preprocessing_layers=preprocessing_layers,
#     fc_layer_params=actor_fc_layer_params,)

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
    td_errors_loss_fn=loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)

ddpg_step_list, ddpg_reward_list, ddpg_name, ddpg_loss_step, ddpg_loss_list, ddpg_train_returns=train_and_evaluate_ACagent(tf_agent, train_env, eval_env, num_iterations=num_iterations, batch_size=batch_size, name='ddpg')

# from now on, we will implement ppo agent
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.networks import actor_distribution_network

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    preprocessing_layers=preprocessing_layers,
    fc_layer_params=actor_fc_layer_params,)

value_net = ValueNetwork(train_env.observation_spec())

global_step = tf.compat.v2.Variable(0)

tf_agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=actor_optimizer,
    train_step_counter=global_step)


ppo_step_list, ppo_reward_list, ppo_name, ppo_loss_step, ppo_loss_list, ppo_train_returns=train_and_evaluate_ACagent(tf_agent, train_env, eval_env, num_iterations=ppo_num_iterations, batch_size=batch_size, name='ppo')

# from now on, sac agent will be implemented

from tf_agents.agents.sac import sac_agent
from tf_agents.networks import actor_distribution_network, normal_projection_network
from tf_agents.agents.ddpg import critic_network

critic_net = critic_network.CriticNetwork(
    (train_env.observation_spec(),  train_env.action_spec()),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params)

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    preprocessing_layers=preprocessing_layers,
    fc_layer_params=actor_fc_layer_params,)

global_step = tf.compat.v2.Variable(0)

tf_agent = sac_agent.SacAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=actor_optimizer,
    critic_optimizer=critic_optimizer,
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)

sac_step_list, sac_reward_list, sac_name, sac_loss_step, sac_loss_list, sac_train_returns=train_and_evaluate_ACagent(tf_agent, train_env, eval_env, num_iterations=num_iterations, batch_size=batch_size, name='sac')

# from now on, td3_agent will be implemented

from tf_agents.agents.td3 import td3_agent

# when I implemented ActorDistributionNetwork, I got error... I have to solve it.

actor_net=CustomActorNetwork(train_env.observation_spec(),
train_env.action_spec(),
preprocessing_layers=preprocessing_layers,
fc_layer_params=actor_fc_layer_params)

# actor_net = actor_distribution_network.ActorDistributionNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     preprocessing_layers=preprocessing_layers,
#     fc_layer_params=actor_fc_layer_params,)

tf_agent=td3_agent.Td3Agent(
train_env.time_step_spec(),
train_env.action_spec(),
actor_network=actor_net,
critic_network=critic_net,
actor_optimizer=actor_optimizer,
critic_optimizer=critic_optimizer,
target_update_tau=target_update_tau,
target_update_period=target_update_period,
td_errors_loss_fn=loss_fn,
gamma=gamma,
reward_scale_factor=reward_scale_factor,
gradient_clipping=gradient_clipping,
train_step_counter=global_step)

td_step_list, td_reward_list, td_name, td_loss_step, td_loss_list, td_train_returns=train_and_evaluate_ACagent(tf_agent, train_env, eval_env, num_iterations=num_iterations, batch_size=batch_size, name='td3Agent')

from vt import plot_compare

loss_list=[[ddpg_loss_list, 'ddpg'], [sac_loss_list, 'sac'], [ppo_loss_list, 'ppo'], [td_loss_list, 'td3']]
reward_list=[[ddpg_reward_list, 'ddpg'], [sac_reward_list, 'sac'], [ppo_reward_list, 'ppo'], [td_reward_list, 'td3']]
train_reward_list=[[ddpg_train_returns, 'ddpg'], [sac_train_returns, 'sac'], [ppo_train_returns, 'ppo'], [td_train_returns, 'td3']]

plot_compare(td_loss_step, loss_list, name='loss')
plot_compare(td_step_list, reward_list, name='reward')
plot_compare(td_step_list, train_reward, name='reward in train step')
