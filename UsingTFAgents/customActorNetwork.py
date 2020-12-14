import numpy as np
import tensorflow as tf
from tf_agents.networks import network, utils
from tf_agents.networks import encoding_network
from tf_agents.specs import array_spec
from tf_agents.utils import nest_utils
from tf_agents.utils import common as common_utils
from tf_agents.policies import actor_policy, ou_noise_policy
import tf_agents.trajectories.time_step as ts

class CustomActorNetwork(network.Network):
    def __init__(self, observation_spec, action_spec, preprocessing_layers=None,
    preprocessing_combiner=None, conv_layer_params=None, fc_layer_params=(64, 64),
    dropout_layer_params=None, activation_fn=tf.keras.activations.relu, enable_last_layer_zero_initializer=False,
    name='CustomActorNetwork'):
        super().__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)

        self._action_spec=action_spec
        flat_action_spec=tf.nest.flatten(action_spec)
        self._single_action_spec=flat_action_spec[0]
        kernel_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        # kernel_initializer=tf.keras.initializers.VarianceScaling(
        # scale=1./3., mode='fan_in', distribution='uniform')

        self._encoder=encoding_network.EncodingNetwork(
        observation_spec, preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        fc_layer_params=fc_layer_params,
        dropout_layer_params=dropout_layer_params,
        activation_fn=activation_fn,
        kernel_initializer=kernel_initializer,
        batch_squash=False)

        initializer=tf.keras.initializers.RandomUniform(-0.001, 0.001)

        self._action_projection_layer=tf.keras.layers.Dense(
        flat_action_spec[0].shape.num_elements(), activation=tf.keras.activations.tanh, kernel_initializer=initializer, name='action')

    def call(self, observations, step_type=(), network_state=(), training=False, mask=None):
        outer_rank=nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash=utils.BatchSquash(outer_rank)
        observations=tf.nest.map_structure(batch_squash.flatten, observations)

        state, network_state=self._encoder(
        observations, step_type=step_type, network_state=network_state
        )

        actions=self._action_projection_layer(state)
        actions=common_utils.scale_to_spec(actions, self._single_action_spec)
        actions=batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state

# Experiment

if __name__=='__main__':
    # observation_spec={
    # 'state_1': array_spec.BoundedArraySpec((10, ), np.float32, -10, 10),
    # 'state_2': array_spec.BoundedArraySpec((10, ), np.float32, -10, 10)
    # }
    observation_spec=array_spec.BoundedArraySpec((20, ), np.float32, -10, 10)
    action_spec=array_spec.BoundedArraySpec((1, ), np.float32, -1, 1)

    from tf_agents.environments import random_py_environment, tf_py_environment

    random_env=random_py_environment.RandomPyEnvironment(observation_spec, action_spec)

    tf_env=tf_py_environment.TFPyEnvironment(random_env)

    time_step=tf_env.reset()
    print('time_step is', time_step)
    preprocessing_layers=tf.keras.models.Sequential(
    [tf.keras.layers.Reshape((20, 1)),
    tf.keras.layers.Conv1D(10, 5),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Flatten()
    ])
    # preprocessing_layers={'state_1': tf.keras.layers.Dense(5), 'state_2': tf.keras.layers.Dense(10)}
    # preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1)

    # investigate actor!
    actor=CustomActorNetwork(tf_env.observation_spec(), tf_env.action_spec(), preprocessing_layers=preprocessing_layers)

    network_state=()
    # print('time_step.observation is', time_step.observation)
    #
    # outer_rank=nest_utils.get_outer_rank(time_step.observation, actor.input_tensor_spec)
    # print('actor.input_tensor_spec', actor.input_tensor_spec)
    # print("outer_rank=nest_utils.get_outer_rank(time_step.observation, actor.input_tensor_spec)", outer_rank)
    # batch_squash=utils.BatchSquash(outer_rank)
    # print("batch_squash", batch_squash)
    #
    # observation=tf.nest.map_structure(batch_squash.flatten, time_step.observation)
    # print("observations=tf.nest.map_structure(batch_squash.flatten, observations)", observation)
    #
    # state, network_state=actor._encoder(
    # time_step.observation, step_type=time_step.step_type, network_state=network_state)
    #
    # actions=actor._action_projection_layer(state)
    # print("actions=actor._action_projection_layer(state) :", actions)
    #
    # actions=common_utils.scale_to_spec(actions, actor._single_action_spec)
    # print('actor._single_action_spec', actor._single_action_spec)
    #
    # print("actions=common_utils.scale_to_spec(actions, self._single_action_spec)", actions)
    #
    # actions=batch_squash.unflatten(actions)
    # print("actions :", actions)
    #
    # # print('actor"s state, actor"s network state', state, network_state)
    # print("actor's action_spec is", actor._action_spec)
    # print("tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state", tf.nest.pack_sequence_as(actor._action_spec, [actions]))
    # print('tf.nest.flatten(action_spec)', tf.nest.flatten(action_spec))
    state=tf.constant(1.0, shape=(10, ))
    # input_tensor_spec = ([tf.TensorSpec(3)] * 2, [tf.TensorSpec(3)] * 5)

    print(actor(time_step.observation, time_step.step_type))

    time_step_spec=ts.time_step_spec(time_step.observation, time_step.reward)
    print('time_step.observation is ', time_step.observation)
    print(time_step_spec)

    policy=actor_policy.ActorPolicy(time_step_spec=time_step_spec, action_spec=action_spec, actor_network=actor, clip=True)
    action_step=policy.action(time_step)
    print('action step is', action_step)

    collect_policy = actor_policy.ActorPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec,
        actor_network=actor, clip=False)

    from tf_agents.policies.policy_saver import PolicySaver

    weights=actor.variables

    # print(weights)
    # print(type(weights))
    # print(len(weights))
    # tensor=tf.zeros_like(weights.shape)
    # print(tensor)
    # actor.get_initial_state(tensor)
    # print(actor.variables)

    # collect_policy = ou_noise_policy.OUNoisePolicy(
    #     collect_policy,
    #     clip=True)
