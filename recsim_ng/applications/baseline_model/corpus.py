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

"""Corpus entity for partially observable RL simulation."""
import edward2 as ed  # type: ignore
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.recommendation import corpus
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import pandas as pd
import ast

Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space

@gin.configurable
class CorpusWithEmbeddingsAndTopics(corpus.Corpus):
  """Defines a corpus wiht static topics and embeddings."""
  def __init__(self,
               config, data_path, col_name_topic, col_name_embed):

    super().__init__(config)
    self._data_path = data_path
    self._num_users = config['num_users']
    self._doc_embed_dim = config['doc_embed_dim']
    self._col_name_topic = col_name_topic
    self._col_name_embed = col_name_embed
  
  def initial_state(self):
    df = pd.read_csv(self._data_path)
    doc_features = tf.convert_to_tensor([ast.literal_eval(embed) for embed in df[self._col_name_embed]])
    doc_topic = tf.convert_to_tensor(df[self._col_name_topic])
    topic_quality_means = tf.random.uniform([self._num_topics], minval=-1.0, maxval=1.0)
    doc_quality_var = 0.1
    doc_quality = ed.Normal(
        loc=tf.gather(topic_quality_means, doc_topic), scale=doc_quality_var)
    doc_recommend_times = tf.zeros([self._num_users, self._num_docs])
    doc_click_times = tf.zeros([self._num_users, self._num_docs])

    return Value(
        # doc_id=0 is reserved for "null" doc.
        doc_id=ed.Deterministic(
            loc=tf.range(start=1, limit=self._num_docs + 1, dtype=tf.int32)),
        doc_topic=doc_topic,
        doc_quality=doc_quality,
        doc_features=doc_features,
        doc_recommend_times = doc_recommend_times,
        doc_click_times = doc_click_times
    )

  def next_state(self, previous_state, user_response, slate_docs):
    new_doc_recommend_times = []
    doc_id_recommend = slate_docs.get("doc_id") # user, slate_size
    
    chosen_idx = list(user_response.get("choice"))
    doc_id_click = [] # user, 1
    for user_idx, user_choice in enumerate(chosen_idx):
      doc_id_click.append(doc_id_recommend[user_idx][user_choice])

    new_doc_click_times = []
    for user_idx, doc_id in enumerate(doc_id_recommend):
      each_user = []
      for doc_index, times in enumerate(previous_state.get("doc_recommend_times")[user_idx]):
        if (doc_index in doc_id-1):
          each_user.append(times+1)
        else: each_user.append(times)
      new_doc_recommend_times.append(each_user)
    new_doc_recommend_times = tf.convert_to_tensor(new_doc_recommend_times)

    for user_idx, doc_id in enumerate(doc_id_click):
      each_user = []
      for doc_index, times in enumerate(previous_state.get("doc_click_times")[user_idx]):
        if (doc_index == doc_id-1):
          each_user.append(times+1)
        else: each_user.append(times)
      new_doc_click_times.append(each_user)
    new_doc_click_times = tf.convert_to_tensor(new_doc_click_times)

    return Value(
        # doc_id=0 is reserved for "null" doc.
        doc_id=previous_state.get("doc_id"),
        doc_topic=previous_state.get("doc_topic"),
        doc_quality=previous_state.get("doc_quality"),
        doc_features=previous_state.get("doc_features"),
        doc_recommend_times = new_doc_recommend_times,
        doc_click_times = new_doc_click_times
    )
      

  def available_documents(self, corpus_state):
    return corpus_state.map(tf.identity)
  
  def specs(self):
    state_spec = ValueSpec(
        doc_id=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_docs, dtype=tf.int32)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_topics, dtype=tf.int32)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones(self._num_docs) * -np.Inf,
                high=np.ones(self._num_docs) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.ones((self._num_docs, self._doc_embed_dim)) * np.Inf,
                high=np.ones((self._num_docs, self._doc_embed_dim)))),
       #Notice: each user needs a isolated space for their click and recommended history, so the shape is (num_users, num_docs)
        doc_recommend_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=tf.int32)), 
        doc_click_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=tf.int32)),)
    return state_spec.prefixed_with("state").union(
        state_spec.prefixed_with("available_docs"))



@gin.configurable
class CorpusWithTopicAndQuality(corpus.Corpus):
  """Defines a corpus with static topic and quality distributions."""

  def __init__(self,
               config,
               topic_min_utility = -1.,
               topic_max_utility = 1.,
               video_length = 2.):

    super().__init__(config)
    self._topic_min_utility = topic_min_utility
    self._topic_max_utility = topic_max_utility
    self._video_length = video_length

  def initial_state(self):
    """The initial state value."""
    # 70% topics are trashy, rest are nutritious.
    num_trashy_topics = int(self._num_topics * 0.7)
    num_nutritious_topics = self._num_topics - num_trashy_topics
    trashy = tf.linspace(self._topic_min_utility, 0., num_trashy_topics)
    nutritious = tf.linspace(0., self._topic_max_utility, num_nutritious_topics)
    topic_quality_means = tf.concat([trashy, nutritious], axis=0)
    # Equal probability of each topic.
    doc_topic = ed.Categorical(
        logits=tf.zeros((self._num_docs, self._num_topics)), dtype=tf.int32)
    # Fixed variance for doc quality.
    doc_quality_var = 0.1
    doc_quality = ed.Normal(
        loc=tf.gather(topic_quality_means, doc_topic), scale=doc_quality_var)
    # 1-hot doc features.
    doc_features = ed.Normal(
        loc=tf.one_hot(doc_topic, depth=self._num_topics), scale=0.7)
    # All videos have same length.
    video_length = ed.Deterministic(
        loc=tf.ones((self._num_docs,)) * self._video_length)

    return Value(
        # doc_id=0 is reserved for "null" doc.
        doc_id=ed.Deterministic(
            loc=tf.range(start=1, limit=self._num_docs + 1, dtype=tf.int32)),
        doc_topic=doc_topic,
        doc_quality=doc_quality,
        doc_features=doc_features,
        doc_length=video_length)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    del user_response
    del slate_docs
    return previous_state.map(ed.Deterministic)

  def available_documents(self, corpus_state):
    """The available_documents value."""
    return corpus_state.map(ed.Deterministic)

  def specs(self):
    state_spec = ValueSpec(
        doc_id=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_docs)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_topics)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones(self._num_docs) * -np.Inf,
                high=np.ones(self._num_docs) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
        doc_length=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * np.Inf)))
    return state_spec.prefixed_with("state").union(
        state_spec.prefixed_with("available_docs"))


class StaticCorpus(CorpusWithTopicAndQuality):
  """Defines a static corpus with state passed from outside."""

  def __init__(self, config, static_state):
    super().__init__(config)
    self._static_state = static_state

  def initial_state(self):
    """The initial state value."""
    return self._static_state.map(tf.identity)

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    del user_response
    del slate_docs
    return previous_state.map(tf.identity)
