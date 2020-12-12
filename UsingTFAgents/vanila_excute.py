from StockEnvironment import StockEnv
from datapreprocess import StatesPreprocess
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf

from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from Evaluations import compute_avg_return

tf.compat.v1.enable_v2_behavior()

# Data preprocess

datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'

train_states, eval_states=StatesPreprocess(datadir, reward_length=1)

# Hyperparameters

num_iterations=1
collect_episodes_per_iteration=2
replay_buffer_capacity=1000
fc_layer_params=(64, 64)

learning_rate=1e-3
log_interval=50
num_eval_episodes=1
eval_interval=100
delay=True

# Environment

train_env=StockEnv(train_states, discrete=False, delay=delay)
eval_env=StockEnv(eval_states, discrete=False, delay=delay)

train_env=tf_py_environment.TFPyEnvironment(train_env)
eval_env=tf_py_environment.TFPyEnvironment(eval_env)

# Agent

print('observation spec', train_env.observation_spec())
print('action_spec', train_env.action_spec())

actor_net=actor_distribution_network.ActorDistributionNetwork(
train_env.observation_spec(),
train_env.action_spec(),
fc_layer_params=fc_layer_params
)

optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

train_step_counter=tf.compat.v2.Variable(0)

tf_agent=reinforce_agent.ReinforceAgent(
train_env.time_step_spec(),
train_env.action_spec(),
actor_network=actor_net,
optimizer=optimizer,
normalize_returns=False,
train_step_counter=train_step_counter
)

# print(dir(tf_agent))
# print(tf_agent.trainable_variables==tf_agent._actor_network.trainable_variables)
# print(zxc)

tf_agent.initialize()

# Policies

eval_policy=tf_agent.policy
collect_policy=tf_agent.collect_policy

# Replay buffer

batch_size=1
replay_buffer=tf_uniform_replay_buffer.TFUniformReplayBuffer(
data_spec=tf_agent.collect_data_spec,
batch_size=batch_size,
max_length=replay_buffer_capacity
)

# Data collection

time_step=train_env.current_time_step()
action_step=tf_agent.collect_policy.action(time_step)
next_time_step=train_env.step(action_step.action)
traj=trajectory.from_transition(time_step, action_step, next_time_step)

print('time step: ', time_step)
print('action step: ', action_step)
print('next time step: ', next_time_step)
print('trajectory: ', traj)
print('trajector"s boundary', traj.is_boundary())


def collect_episode(env, policy, num_episodes):
    episode_counter=0
    env.reset()

    while episode_counter<num_episodes:
        time_step=env.current_time_step()
        action_step=policy.action(time_step)
        next_time_step=env.step(action_step.action)
        traj=trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter+=1

collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)

# Train the agent
tf_agent.train=common.function(tf_agent.train)

tf_agent.train_step_counter.assign(0)

avg_return=compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
returns=[avg_return]


num_iterations=1000
for _ in range(num_iterations):
    collect_episode(train_env, tf_agent.collect_policy, collect_episodes_per_iteration)
    before=tf_agent._actor_network.trainable_variables
    before=tf_agent.trainable_variables

    experience=replay_buffer.gather_all()
    train_loss=tf_agent.train(experience)
    replay_buffer.clear()
    step=tf_agent.train_step_counter.numpy()

    if step % log_interval==0:
        print('step = {0}: loss = {1}'.format(step, train_loss.loss))

    if step % eval_interval == 0:
        avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim(top=1)
# plt.show()
