from tf_agents.agents.td3 import td3_agent

'''
actor_net=CustomActorNetwork(train_env.observation_spec(),
train_env.action_spec(), preprocessing_layers=preprocessing_layers,
fc_layer_params=actor_fc_layer_params)

loss_fn=tf.math.squared_difference
loss_fn=tf.compat.v1.losses.mean_squared_error
loss_fn=tf.keras.losses.KLDivergence

tf_agent=td3_agent.Td3Agent(
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
td_errors_loss_fn=loss_fn,
gamma=gamma,
reward_scale_factor=reward_scale_factor,
gradient_clipping=gradient_clipping,
train_step_counter=global_step)
'''

if __name__=='__main__':
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
    num_iterations=100
    global_step = tf.compat.v2.Variable(0)
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
    train_env.action_spec(),
    preprocessing_layers=preprocessing_layers,
    fc_layer_params=actor_fc_layer_params)

    critic_net = critic_network.CriticNetwork(
        (train_env.observation_spec(),  train_env.action_spec()),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params)

    tf_agent=td3_agent.Td3Agent(
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
    td_errors_loss_fn=loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    train_step_counter=global_step)
    
    td_step_list, td_reward_list, td_name=train_and_evaluate_ACagent(tf_agent, train_env, eval_env, num_iterations=num_iterations, batch_size=batch_size, name='td3Agent')
