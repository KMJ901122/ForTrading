import tensorflow as tf
import numpy as np
from tf_agents.environments import py_environment, tf_environment, tf_py_environment, utils, wrappers, suite_gym
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class StockEnv(py_environment.PyEnvironment):
    def __init__(self, states, discrete=True): # states will be price changes.
        self.window=len(states[0][:-1]) # last element will be a reward.
        self.states=states
        self.time_flow=0
        self.total_reward=0
        self.threshold=0.7
        self.curr_position=0

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
        self.total_reward=0
        self._state = self.states[0][:-1]
        self._episode_ended = False
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action):
        self._state=self.states [self.time_flow][:-1]
        reward=self.states[self.time_flow][-1]*action
        self.total_reward+=tf.divide(reward, tf.abs(action))
        self.time_flow+=1
        if self.time_flow==self.states.shape[0]:
            termination=ts.termination(np.array(self._state, dtype=np.float32), reward)
            self.reset()
            return termination
        else:
            return ts.transition(
              np.array(self._state, dtype=np.float32), reward, discount=1.0)
