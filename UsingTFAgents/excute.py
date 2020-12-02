from __future__ import absolute_import
from StockEnvironment import StockEnv
from datapreprocess import StatesPreprocess
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from vt import v_plot
import tensorflow as tf
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec, tensor_spec
from tf_agents import specs
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent, tanh_normal_projection_network
from tf_agents.experimental.train import actor, learner, triggers
from tf_agents.experimental.train.utils import spec_utils, strategy_utils, train_utils
from tf_agents.metrics import py_metric, tf_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.policies import greedy_policy, py_tf_eager_policy, random_py_policy
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer, py_uniform_replay_buffer


datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'
data=pd.read_csv(datadir)
train_states, eval_states=StatesPreprocess(datadir, reward_length=20)
v_plot(data)
print(abc)
# Hyperparameters

num_iterations=10
initial_collect_steps=1000
collect_steps_per_iteration=1
replay_buffer_capacity=10000

batch_size=16


critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 50 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 100 # @param {type:"integer"}

policy_save_interval = 1000 # @param {type:"integer"}

# Environment

discrete=False # continuous action 1-> utilize whole asset for buying, 0.5 half of whole asset...

collect_env=StockEnv(train_states, discrete)
eval_env=StockEnv(eval_states, discrete)

collect_env=tf_py_environment.TFPyEnvironment(collect_env)
eval_env=tf_py_environment.TFPyEnvironment(eval_env)

# Agent

observation_spec, action_spec, time_step_spec=(spec_utils.get_tensor_specs(collect_env))
print('observation spec: ', observation_spec)
print('action spec: ', action_spec)
print('time step spec:', time_step_spec)

use_gpu=True
strategy=strategy_utils.get_strategy(tpu=False, use_gpu=use_gpu)

with strategy.scope():
    critic_net=critic_network.CriticNetwork((observation_spec, action_spec),
    observation_fc_layer_params=None,
    action_fc_layer_params=None,
    joint_fc_layer_params=critic_joint_fc_layer_params,
    kernel_initializer='glorot_uniform',
    last_kernel_initializer='glorot_uniform')

with strategy.scope():
    actor_net=actor_distribution_network.ActorDistributionNetwork(
    observation_spec, action_spec,
    fc_layer_params=actor_fc_layer_params,
    continuous_projection_net=(tanh_normal_projection_network.TanhNormalProjectionNetwork)
    )

with strategy.scope():
    train_step=train_utils.create_train_step()

    tf_agent=sac_agent.SacAgent(
    time_step_spec,
    action_spec,
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=critic_learning_rate),
    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.math.squared_difference,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    train_step_counter=train_step
    )

    tf_agent.initialize()

# Replay buffer

# TF
replay_buffer=tf_uniform_replay_buffer.TFUniformReplayBuffer(data_spec=tf_agent.collect_data_spec,
batch_size=batch_size,
max_length=replay_buffer_capacity)

# Py
# replay_buffer=py_uniform_replay_buffer.PyUniformReplayBuffer(capacity=replay_buffer_capacity,
# data_spec=tensor_spec.to_nest_array_spec(tf_agent.collect_data_spec))

print('tf agent coolect data spec', tf_agent.collect_data_spec)
print('tf agent collect data spec fields', tf_agent.collect_data_spec._fields)

# Policies

tf_eval_policy=tf_agent.policy
eval_policy=py_tf_eager_policy.PyTFEagerPolicy(tf_eval_policy, use_tf_function=True)

tf_collect_policy=tf_agent.collect_policy
collect_policy=py_tf_eager_policy.PyTFEagerPolicy(tf_collect_policy, use_tf_function=True)

random_policy=random_py_policy.RandomPyPolicy(collect_env.time_step_spec(), collect_env.action_spec())

# Actors

replay_observer=[replay_buffer.add_batch]
initial_collect_actor=actor.Actor(
collect_env, random_policy, train_step, steps_per_run=initial_collect_steps, observers=replay_observer)

initial_collect_actor.run()



# Data collection

# def collect_step(environment, policy, buffer):
#     time_step=environment.current_time_step()
#     action_step=policy.action(time_step)
#     next_time_step=environment.step(action_step.action)
#     traj=trajectory.from_transition(time_step, action_step, next_time_step)
#
#     # Add trajectory to the replay buffer
#     buffer.add_batch(traj)
#
# def collect_data(env, policy, buffer, steps):
#     for _ in range(steps):
#         collect_step(env, policy, buffer)
#
# collect_data(collect_env, )

# ------------ later-------
# time_step=tf_env.reset()
# reward=[]
# steps=[]
# num_episodes=5
#
#
# for _ in range(num_episodes):
#     episode_reward=0
#     episode_steps=0
#     while not time_step.is_last():
#         action=tf.random.uniform((1, ), -1, 1, dtype=tf.int32)
#         time_step=tf_env.step(action)
#         episode_steps+=1
#         episode_reward+=time_step.reward.numpy()
#
#     reward.append(episode_reward)
#     steps.append(episode_steps)
#     time_step=tf_env.reset()
#
# num_steps=np.sum(steps)
# avg_length=np.mean(steps)
# avg_reward=np.mean(reward)
#
# print('num_episodes: ', num_episodes, 'num_steps', num_steps)
# print('avg_length', avg_length, 'avg_reward:', avg_reward)
