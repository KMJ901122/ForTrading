import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class StockEnv(py_environment.PyEnvironment):
    def __init__(self, states, discrete=True, delay=False): # states will be price changes.
        self.window=len(states[0][:-1]) # last element will be a reward.
        self.states=states
        self.time_flow=0
        self.total_reward=1. # 100% asset
        self.action_counter=0
        self.curr_position=0
        self.delay=delay
        self.delay_counter=0
        self.delay_threshold=10

        if delay:
            print('Action delay implemented')
            print('Default delay threshold is ', self.delay_threshold)

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
        self._state = self.states[0][:-1]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        self._state=self.states [self.time_flow][:-1]
        raw_reward=self.states[self.time_flow][-1]
        raw_reward=1.+raw_reward
        reward=raw_reward*action # for actor update just let it raw, anyway it will goes to 1 after long iterations
        # Actually, this is a prediction model if we do not change raw reward.

        if self.delay:
            if self.delay_counter<self.delay_threshold:
                self.delay_counter+=1
                self.total_reward*=raw_reward*self.curr_position
            else:
                if self.curr_position==tf.sign(action): # action not changed
                    self.total_reward*=raw_reward*self.curr_position
                else: # action changed; so delay counter reset
                    self.delay_counter=0
                    self.total_reward*=tf.divide(reward, tf.abs(action))

        else:
            self.total_reward*=tf.divide(reward, tf.abs(action))

        self.time_flow+=1

        if self.curr_position*action<0: # position changed
            self.action_counter+=1

        self.curr_position=tf.sign(action) # current position changed

        if self.time_flow==self.states.shape[0]:
            termination=ts.termination(np.array(self._state, dtype=np.float32), reward)
            print('action counter is ', self.action_counter)
            self.reset()
            return termination
        else:
            return ts.transition(np.array(self._state, dtype=np.float32), reward, discount=1.0)
