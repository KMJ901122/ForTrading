from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks.value_network import ValueNetwork

actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=actor_fc_layer_params)

value_net = ValueNetwork(
    train_env.observation_spec()
)

global_step = tf.compat.v2.Variable(0)
tf_agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    actor_net=actor_net,
    value_net=value_net,
    optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=actor_learning_rate),
    train_step_counter=global_step)
