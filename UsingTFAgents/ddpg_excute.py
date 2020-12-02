from StockEnvironment import StockEnv
from datapreprocess import StatesPreprocess
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
factor=1
ddpg_num_iterations=int(factor*100000)
policy_num_iterations=10

ddpg_initial_collect_steps=int(factor*1000)
collect_steps_per_iteration=1
collect_episodes_per_iteration=2
ddpg_replay_buffer_capacity=int(factor*1000)
policy_replay_buffer_capacity=2000

batch_size = 64  # @param {type:"integer"}

critic_learning_rate = 3e-4  # @param {type:"number"}
actor_learning_rate = 3e-4  # @param {type:"number"}
alpha_learning_rate = 3e-4  # @param {type:"number"}
target_update_tau = 0.005  # @param {type:"number"}
policy_learning_rate = 1e-3  # @param {type:"number"}
target_update_period = 1  # @param {type:"number"}
gamma = 0.99  # @param {type:"number"}
reward_scale_factor = 1.0  # @param {type:"number"}
gradient_clipping = None  # @param

actor_fc_layer_params = (64, 64)
critic_joint_fc_layer_params = (64, 64)

ddpg_log_interval = int(factor * 5000)  # @param {type:"integer"}
policy_log_interval = 25  # @param {type:"integer"}

num_eval_episodes = 3  # @param {type:"integer"}
ddpg_eval_interval = int(factor * 10000)  # @param {type:"integer"}
policy_eval_interval = 50  # @param {type:"integer"}

datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'
train_states, eval_states=StatesPreprocess(datadir, reward_length=10)

train_py_env=StockEnv(train_states, discrete=False)
eval_py_env=StockEnv(eval_states, discrete=False)

train_env=tf_py_environment.TFPyEnvironment(train_py_env)
eval_env=tf_py_environment.TFPyEnvironment(eval_py_env)

def train_and_evaluate_agent_ddpg(tf_agent, name='agent'):
    tf_agent.initialize()

    eval_policy=greedy_policy.GreedyPolicy(tf_agent.policy)
    collect_policy=gaussian_policy.GaussianPolicy(tf_agent.collect_policy)

    replay_buffer=tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=ddpg_replay_buffer_capacity
    )

    initial_collect_driver=dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=ddpg_initial_collect_steps
    )

    initial_collect_driver.run()


    collect_driver=dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration
    )

    tf_agent.train=common.function(tf_agent.train)
    collect_driver.run=common.function(collect_driver.run)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evalute the agent's policy once before training
    avg_return=compute_avg_return(eval_env, eval_policy, num_eval_episodes)
    returns=[(0, avg_return)]

    for _ in range(ddpg_num_iterations):
        # Collect a few steps using collect_policy and svae to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_driver.run()

        dataset=replay_buffer.as_dataset(batch_size, 2)
        iterator=iter(dataset)
        # Sample a batch of data from the buffer and update the agent's network
        experience, _ =next(iterator)
        train_loss=tf_agent.train(experience)

        step=tf_agent.train_step_counter.numpy()

        if step % ddpg_log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % ddpg_eval_interval == 0:
            avg_return = compute_avg_return(
                eval_env, eval_policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append((step, avg_return))

    steps_list = [r[0] for r in returns]
    rewards_list = [r[1] for r in returns]
    return steps_list, rewards_list

    # plt.plot(steps_list, rewards_list)
    # plt.ylabel('Average Return')
    # plt.xlabel('Step')
    # plt.title(name)
    # plt.show()

from tf_agents.agents.ddpg import ddpg_agent, actor_network, critic_network

actor_net = actor_network.ActorNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=actor_fc_layer_params)

print('train_env.observation_spec', train_env.observation_spec())

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
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate),
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)

# train_and_evaluate_agent_ddpg(tf_agent, name='ddpg')
