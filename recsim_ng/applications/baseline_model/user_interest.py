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
"""User entity from the recsys_partially_obervable_rl."""
"""User entity for long-term interests evolution simulation."""
from typing import Callable, Optional, Sequence

import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import affinities as affinity_lib
from recsim_ng.entities.choice_models import selectors as selector_lib
from recsim_ng.entities.recommendation import user
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import state
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp
import random
import edward2 as ed

tfd = tfp.distributions
Constructor = Callable[Ellipsis, object]
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


def tensor_space(low = -np.Inf,
                 high = np.Inf,
                 shape = ()):
  return Space(spaces.Box(low=low, high=high, shape=shape))

class UserWithInterestedTopics(object):
  """Defines User with interested topics. """

  def __init__(self, config):
    self._slate_size = config['slate_size']
    self._num_topics = config['num_topics']
    self._num_users = config['num_users']
    # The more interested topics means the number interested topics is larger than the slate size
    self._more_interested_topics = config['more_interested_topics']
  
  def initial_state(self):
    if self._more_interested_topics:
      interested_topics_num = int(self._slate_size*1.5)
    else:
      interested_topics_num = int(self._slate_size*0.5)
    
    all_users_interested_topics = []
    for i in range(self._num_users):
      interested_topics = np.zeros(self._num_topics, dtype=np.float32)
      interested_topics[np.random.choice(self._num_topics, interested_topics_num, replace=False)] = 1
      all_users_interested_topics.append(interested_topics)
    # Set the mean value of each user's interest to be the all_users_interested_topics
    interest_initial_state = Value(state = ed.Normal(loc=tf.convert_to_tensor(all_users_interested_topics), scale=0.5*tf.ones([self._num_users, self._num_topics])))
    print("User interest: ",interest_initial_state.get('state'))
    return interest_initial_state

@gin.configurable
class InterestEvolutionUser(user.User):
  """Dynamics of a user whose interests evolve over time."""

  def __init__(
      self,
      config,
      affinity_model_ctor = affinity_lib.TargetPointSimilarity,
      choice_model_ctor = selector_lib.MultinomialLogitChoiceModel,
      no_click_mass = 0.,
      # Step size for updating user interests based on consumed documents
      # (small!). We may want to have different values for different interests
      # to represent how malleable those interests are, e.g., strong dislikes
      # may be less malleable).
      interest_step_size = 0.1,
      reset_users_if_timed_out = False,
      interest_update_noise_scale = None,
      initial_interest_generator:UserWithInterestedTopics = None,
      max_user_affinity = 10.0):
    super().__init__(config)
    self._config = config
    self._doc_embed_dim = config['doc_embed_dim']
    self._num_topics = config['num_topics']
    self._max_user_affinity = max_user_affinity
    self._affinity_model = affinity_model_ctor(
        (self._num_users,), config['slate_size'], 'negative_cosine')
    self._choice_model = choice_model_ctor(
        (self._num_users,), no_click_mass * tf.ones(self._num_users))
    self._interest_generator = initial_interest_generator
    if interest_update_noise_scale is None:
      interest_noise = None
    else:
      interest_noise = interest_update_noise_scale * tf.ones(
          self._num_users, dtype=tf.float32)
    interest_model = dynamic.ControlledLinearScaledGaussianStateModel(
        dim=self._num_topics,
        transition_scales=None,
        control_scales=interest_step_size *
        tf.ones(self._num_users, dtype=tf.float32),
        noise_scales=interest_noise,
        # initial_dis_scales is the standard deviation of the initial distribution
        initial_dist_scales=tf.ones(self._num_users, dtype=tf.float32))
    self._interest_model = dynamic.NoOPOrContinueStateModel(
        interest_model, batch_ndims=1)

  def initial_state(self):
    """The initial state value."""
    if self._interest_generator is not None:
      interest_initial_state = self._interest_generator.initial_state()
    else:
      interest_initial_state = self._interest_model.initial_state()
      print("Without interested topics user: ", interest_initial_state.get('state'))
      interest_initial_state = Value(
          state=tf.identity(interest_initial_state.get('state'))).union(
              interest_initial_state.prefixed_with('linear_update'))
    return interest_initial_state.prefixed_with('interest')

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    chosen_docs = user_response.get('choice')
    chosen_doc_features = selector_lib.get_chosen(slate_docs, chosen_docs)
    # Calculate utilities.
    user_interests = previous_state.get('interest.state')
    doc_vector = chosen_doc_features.get('doc_vector')
    # User interests are increased/decreased towards the consumed document's
    # topic proportinal to the document quality.
    direction = tf.expand_dims(
        chosen_doc_features.get('doc_quality'), axis=-1) * (
            doc_vector - user_interests)
    # TODO: need to remove consumed time, consider other indicators
    linear_update = self._interest_model.next_state(
        previous_state.get('interest'),
        Value(
            input=direction,
            condition=tf.less(tf.random.uniform([self._num_users], minval=-1.0, maxval=1.0), 0.)))
    # We squash the interest vector to avoid infinite blow-up using the function
    # 4 * M * (sigmoid(X/M) - 0.5) which is roughly linear around the origin and
    # softly saturates at +/-2M. These constants are not intended to be tunable.
    next_interest = Value(
        state=4.0 * self._max_user_affinity *
        (tf.sigmoid(linear_update.get('state') / self._max_user_affinity) -
         0.5)).union(linear_update.prefixed_with('linear_update'))
    return next_interest.prefixed_with('interest')

  def observation(self, user_state):
    del user_state
    return Value()

  def next_response(self, previous_state, slate_docs):
    """The response value after the initial value."""
    affinities = self._affinity_model.affinities(  # pytype: disable=attribute-error  # trace-all-classes
        previous_state.get('interest.state'),
        slate_docs.get('doc_vector')).get('affinities')
    choice = self._choice_model.choice(affinities + 2.0)  # pytype: disable=attribute-error  # trace-all-classes
    # chosen_doc_idx = choice.get('choice')
    # Calculate consumption time. Negative quality documents generate more
    # engagement but ultimately lead to negative interest evolution.
    # doc_quality = slate_docs.get('doc_quality')
    # consumed_fraction = tf.sigmoid(-doc_quality)
    # doc_length = slate_docs.get('doc_length')
    # consumed_time = consumed_fraction * 1.0
    # chosen_doc_responses = selector_lib.get_chosen(
    #     Value(consumed_time=consumed_time), chosen_doc_idx)
    return choice

  def specs(self):
    # TODO: interest_spec only need interest + state as prefix
    if self._interest_generator is not None:
      interest_spec = ValueSpec(
        state = tensor_space(low=-10.0, high=10.0, shape=(self._num_users, self._num_topics))
      )
    else:
      interest_spec = ValueSpec(
        state=tensor_space(
            low=-10.0, high=10.0, shape=(
                self._num_users, self._num_topics))).union(
                    self._interest_model.specs().prefixed_with('linear_update'))
    state_spec = interest_spec.prefixed_with('interest')
    # Notice: no more consumed time
    response_spec = self._choice_model.specs()  # pytype: disable=attribute-error  # trace-all-classes
    observation_spec = ValueSpec()
    return state_spec.prefixed_with('state').union(
        observation_spec.prefixed_with('observation')).union(
            response_spec.prefixed_with('response'))
  
class StaticUser(InterestEvolutionUser):
  """Defines a static user with state passed from outside."""

  def __init__(self, config, static_state, interest_generator = None):
    super().__init__(config)
    self._static_state = static_state
    self._interest_generator = interest_generator

  def initial_state(self):
    """The initial state value."""
    return self._static_state.map(tf.identity)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    del user_response
    del slate_docs
    return previous_state.map(tf.identity)


