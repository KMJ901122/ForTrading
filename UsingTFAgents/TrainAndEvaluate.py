from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver, dynamic_episode_driver
from tf_agents.policies import greedy_policy, random_tf_policy, gaussian_policy
from Evaluations import compute_avg_return

def train_and_evaluate_ACagent(tf_agent, train_env=None, eval_env=None, num_iterations=None, batch_size=32, replay_buffer_capacity=1000, name='agent'):

    if train_env is None:
        raise ValueError("train_env is None! Environment should be implemented")

    if eval_env is None:
        raise ValueError("eval_env is None! Environment for evaluation should be implemented")

    if num_iterations is None:
        raise ValueError("Number of iterations should be implemented!")

    tf_agent.initialize()

    initial_collect_steps=1
    collect_steps_per_iteration=1

    print('Initial collect step is', initial_collect_steps)
    print('collect steps per iteration', collect_steps_per_iteration)
    print('batch size is ', batch_size)
    print('replay buffer capacity is', replay_buffer_capacity)

    eval_policy=tf_agent.policy
    collect_policy=gaussian_policy.GaussianPolicy(tf_agent.collect_policy)

    replay_buffer=tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec, batch_size=1, max_length=replay_buffer_capacity)

    initial_collect_driver=dynamic_step_driver.DynamicStepDriver(train_env, collect_policy,
    observers=[replay_buffer.add_batch], num_steps=initial_collect_steps)

    initial_collect_driver.run()

    collect_driver=dynamic_step_driver.DynamicStepDriver(train_env, collect_policy,
    observers=[replay_buffer.add_batch], num_steps=collect_steps_per_iteration)

    tf_agent.train=common.function(tf_agent.train)
    collect_driver.run=common.function(collect_driver.run)

    # Reset the train step
    tf_agent.train_step_counter.assign(0)

    # Evalute the agent's policy once before training
    avg_return=compute_avg_return(eval_env, eval_policy)
    returns=[(0, avg_return)]

    for _ in range(num_iterations):
        # Collect a few steps using collect_policy and svae to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_driver.run()

        dataset=replay_buffer.as_dataset(batch_size, 2)
        iterator=iter(dataset)
        # Sample a batch of data from the buffer and update the agent's network
        experience, _ =next(iterator)
        train_loss=tf_agent.train(experience)

        step=tf_agent.train_step_counter.numpy()

        log_interval=int(num_iterations/20)
        eval_interval=int(num_iterations/10)

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            eval_policy=tf_agent.policy
            avg_return = compute_avg_return(eval_env, eval_policy)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append((step, avg_return))

    steps_list = [r[0] for r in returns]
    rewards_list = [r[1] for r in returns]

    return steps_list, rewards_list, name
