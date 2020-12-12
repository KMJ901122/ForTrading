import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec, tensor_spec
from tf_agents.trajectories import time_step as ts

class StockEnv(py_environment.PyEnvironment):
    def __init__(self, states, discrete=True, delay=False, eval=False): # states will be price changes.
        self.window=len(states[0][:-1]) # last element will be a reward.
        self.states=states
        self.time_flow=0
        self.total_reward=1. # 100% asset
        self.action_counter=0
        self.curr_position=0
        self.delay=delay
        self.delay_counter=0
        self.delay_threshold=5
        self.eval=eval

        if delay:
            if not eval:
                print('Action delay implemented')
                print('Default delay threshold is ', self.delay_threshold)
            else:
                print('Evaluation Environment"s Action delay implemented')
                print('Evaluation Environment"s Default delay threshold is ', self.delay_threshold)
                
        if discrete:
            self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=-1, maximum=1, name='action') # sell hold buy
        else:
            self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1, maximum=1, name='action') # sell hold buy
        self._observation_spec = array_spec.BoundedArraySpec(
        shape=(self.window, ), dtype=np.float32, minimum=-100., maximum=100., name='observation')

        # to combine different models, use 'dict' Type
        # Ex) self._observation_spec=={'state_1': array_spec.BoundedArraySpec(), 'state_2': array_spec.BoundedArraySpec()...}

        self._state = self.states[0][:-1]
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self.time_flow=0
        self.total_reward=1.
        self.action_counter=0
        self.delay_counter=0
        self.curr_position=0
        self._state = self.states[0][:-1]
        self._episode_ended = False

        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        self._state=self.states [self.time_flow][:-1]
        raw_reward=self.states[self.time_flow][-1]
        reward=1.+raw_reward*tf.sign(action)

        if self.delay:
            if self.delay_counter<self.delay_threshold:
                self.delay_counter+=1
                self.total_reward*=1.+raw_reward*self.curr_position
            else:
                if self.curr_position==tf.sign(action): # action not changed
                    self.total_reward*=1.+raw_reward*self.curr_position
                else: # action changed; so delay counter reset
                    self.delay_counter=0
                    self.total_reward*=reward
                    self.curr_position=tf.sign(action)
                    self.action_counter+=1

        else:
            self.total_reward*=reward
            if self.curr_position*tf.sign(action)<0: # position changed
                self.action_counter+=1

            self.curr_position=tf.sign(action) # current position changed

        self.time_flow+=1

        if self.time_flow==self.states.shape[0]:
            termination=ts.termination(np.array(self._state, dtype=np.float32), reward)
            if self.eval==True:
                print('action counter is ', self.action_counter)
                print('total reward during training', self.total_reward)
            self.reset()
            return termination
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward, discount=1.0)

if __name__=='__main__':
    from tf_agents.policies import random_tf_policy, random_py_policy
    from datapreprocess import StatesPreprocess
    from customActorNetwork import CustomActorNetwork
    from tf_agents.policies import actor_policy

    window_size=20
    datadir=r'C:\Users\DELL\Desktop\data\csv\Korea\Samsung.csv'
    train_states, eval_states=StatesPreprocess(datadir, window=window_size, reward_length=10)
    print(eval_states[0])
    print(train_states[0])
    eval_py_env=StockEnv(eval_states, discrete=False, delay=False)
    train_py_env=StockEnv(train_states, discrete=False, delay=False)

    train_env=tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env=tf_py_environment.TFPyEnvironment(eval_py_env)

    time_step = eval_env.reset()
    # print('time_step is ', time_step)
    # input_tensor_spec = tensor_spec.BoundedTensorSpec((20,), tf.float32, minimum=-100, maximum=100)

    time_step_spec=ts.time_step_spec(time_step.observation, time_step.reward)
    action_spec=train_env.action_spec()

    preprocessing_layers=tf.keras.models.Sequential(
    [tf.keras.layers.Reshape((20, 1)),
    tf.keras.layers.Conv1D(10, 5),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Flatten()
    ])

    actor=CustomActorNetwork(train_env.observation_spec(), train_env.action_spec(), preprocessing_layers=preprocessing_layers)
    policy=actor_policy.ActorPolicy(time_step_spec=time_step_spec, action_spec=action_spec, actor_network=actor, clip=True)
    action_step=policy.action(time_step)
    # investigate actor!
    # print(time_step_spec)
    episode_return = 1.

    while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
