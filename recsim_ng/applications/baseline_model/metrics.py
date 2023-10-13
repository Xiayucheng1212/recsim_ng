# coding=utf-8
# Copyright 2022 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Metric Entity from the recsys_partially_observable_rl."""
"""Definitions for recs metrics entities."""
import edward2 as ed  # type: ignore
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.recommendation import metrics
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf

Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space
class SuccessRateMetrics(metrics.RecsMetricsBase):
    
    def __init__(self, config, name='CTRMetrics'):
       super().__init__(config, name)
       self._slate_size=config['slate_size']
       
    def initial_metrics(self):
      return Value(reward = ed.Deterministic(loc=tf.zeros([self._num_users],dtype=tf.float32)),
          cumulative_reward=ed.Deterministic(loc=tf.zeros([self._num_users],dtype=tf.float32)))
      
    def next_metrics(self,previous_metrics, corpus_state,
                   user_state, user_response,
                   slate_doc):
        
        chosen_doc_idx = user_response.get("choice")
        def filter(x):
            return 1.0 if x!=self._slate_size else 0.0
        click=tf.map_fn(filter,chosen_doc_idx,dtype=tf.float32)
        return Value(
        reward=ed.Deterministic(loc=click),
        cumulative_reward=ed.Deterministic(
            loc=previous_metrics.get("cumulative_reward") + click))
        
    def specs(self):
        return ValueSpec(
            reward=Space(
                spaces.Box(
                    low=np.zeros(self._num_users),
                    high=np.array([np.Inf] * self._num_users))),
            cumulative_reward=Space(
                spaces.Box(
                    low=np.zeros(self._num_users),
                    high=np.array([np.Inf] *
                                self._num_users)))).prefixed_with("state")
        
class ClickThroughRateAsRewardMetrics(metrics.RecsMetricsBase):
  """A simple implementation of CTR metrics."""
  def initial_metrics(self):
    return Value(
        reward=ed.Deterministic(loc=tf.zeros([self._num_users])),
        cumulative_reward=ed.Deterministic(loc=tf.zeros([self._num_users])))

  def next_metrics(self, previous_metrics, corpus_state, user_state, user_response, slate_docs):
    del user_state, user_response,slate_docs
    doc_recommend_times = corpus_state.get("doc_recommend_times") + 0.0001
    doc_click_times = corpus_state.get("doc_click_times")
    # Average CTR over each document
    reward = tf.reduce_mean(tf.divide(doc_click_times, doc_recommend_times), axis=1)
    cumulative_reward = previous_metrics.get("cumulative_reward")+reward
    return Value(
      reward = reward, cumulative_reward = cumulative_reward
    )
    
  def specs(self):
    return ValueSpec(
        reward=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.array([np.Inf] * self._num_users))),
        cumulative_reward=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.array([np.Inf] *
                              self._num_users)))).prefixed_with("state")


class ConsumedTimeAsRewardMetrics(metrics.RecsMetricsBase):
  """A minimal implementation of recs metrics."""

  def initial_metrics(self):
    """The initial metrics value."""
    return Value(
        reward=ed.Deterministic(loc=tf.zeros([self._num_users])),
        cumulative_reward=ed.Deterministic(loc=tf.zeros([self._num_users])))

  def next_metrics(self, previous_metrics, corpus_state,
                   user_state, user_response,
                   slate_doc):
    """The metrics value after the initial value."""
    del corpus_state, user_state, slate_doc
    # consumed_time will be -1 for unclicked slates.
    reward = tf.clip_by_value(user_response.get("consumed_time"), 0.0, np.Inf)
    return Value(
        reward=ed.Deterministic(loc=reward),
        cumulative_reward=ed.Deterministic(
            loc=previous_metrics.get("cumulative_reward") + reward))

  def specs(self):
    return ValueSpec(
        reward=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.array([np.Inf] * self._num_users))),
        cumulative_reward=Space(
            spaces.Box(
                low=np.zeros(self._num_users),
                high=np.array([np.Inf] *
                              self._num_users)))).prefixed_with("state")
