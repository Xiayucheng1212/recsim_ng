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
               config):

    super().__init__(config)
    self._data_path = config["data_path"]
    self._col_name_embed = 'embedding'
    self._col_name_topic = 'category_encoded'
    self._num_users = config['num_users']
    self._doc_embed_dim = config['doc_embed_dim']
  
  def initial_state(self):
    df = pd.read_csv(self._data_path+"/embeddings.csv").iloc[:self._num_docs]
    df_vector = pd.read_csv(self._data_path+"/doc_vector_feature.csv").iloc[:self._num_docs]
    #doc_feature is for recommender learning
    doc_features = tf.convert_to_tensor([ast.literal_eval(embed) for embed in df[self._col_name_embed]])
    doc_topic = tf.convert_to_tensor(df[self._col_name_topic])
    #doc_vector is for deciding users click or not
    # doc_vector = ed.Normal(
    #     loc=tf.one_hot(doc_topic, depth=self._num_topics), scale=0.7)
    doc_vector = tf.convert_to_tensor(df_vector.values)
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
        doc_vector = doc_vector,
        doc_recommend_times = doc_recommend_times,
        doc_click_times = doc_click_times
    )

  def next_state(self, previous_state, user_response, slate_docs):
    # Prepare items that needed to be add on 1
    add_doc_recommend_times = np.zeros((self._num_users, self._num_docs))
    doc_id_recommend = slate_docs.get("doc_id") # user, slate_size
    slate_size = doc_id_recommend.shape[1]
    for user_idx, per_user in enumerate(add_doc_recommend_times):
        docs_recommended = doc_id_recommend[user_idx] # slate_size
        per_user[docs_recommended.numpy()-1] = 1.
    add_doc_recommend_times = tf.convert_to_tensor(add_doc_recommend_times, dtype=tf.float32)
    new_doc_recommend_times = tf.add(previous_state.get("doc_recommend_times"), add_doc_recommend_times)

    add_doc_click_times = np.zeros((self._num_users, self._num_docs))
    chosen_idx = user_response.get("choice")
    for user_idx, per_user in enumerate(add_doc_click_times):
      per_user_chosen_idx = chosen_idx[user_idx].numpy() #(1,)
      # User didn't choose any of the slate docs
      if(per_user_chosen_idx == slate_size):
        continue
      # Notice: the doc id starts from 1, however per_user index start from 0.
      docs_clicked = doc_id_recommend[user_idx][per_user_chosen_idx]
      per_user[docs_clicked.numpy()-1] = 1.
    add_doc_click_times = tf.convert_to_tensor(add_doc_click_times, dtype=tf.float32)
    new_doc_click_times = tf.add(previous_state.get("doc_click_times"), add_doc_click_times)

    return Value(
        # doc_id=0 is reserved for "null" doc.
        doc_id=previous_state.get("doc_id"),
        doc_topic=previous_state.get("doc_topic"),
        doc_quality=previous_state.get("doc_quality"),
        doc_features=previous_state.get("doc_features"),
        doc_vector=previous_state.get("doc_vector"),
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
                high=np.ones(self._num_docs) * self._num_docs, dtype=np.int32)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros(self._num_docs),
                high=np.ones(self._num_docs) * self._num_topics, dtype=np.int32)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones(self._num_docs) * -np.Inf,
                high=np.ones(self._num_docs) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.ones((self._num_docs, self._doc_embed_dim)) * np.Inf,
                high=np.ones((self._num_docs, self._doc_embed_dim)))),
        doc_vector=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
       #Notice: each user needs a isolated space for their click and recommended history, so the shape is (num_users, num_docs)
        doc_recommend_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)), 
        doc_click_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)),)
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
    self._num_users = config['num_users']

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
    # video_length = ed.Deterministic(
    #     loc=tf.ones((self._num_docs,)) * self._video_length)
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

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    # Prepare items that needed to be add on 1
    add_doc_recommend_times = np.zeros((self._num_users, self._num_docs))
    doc_id_recommend = slate_docs.get("doc_id") # user, slate_size
    slate_size = doc_id_recommend.shape[1]
    for user_idx, per_user in enumerate(add_doc_recommend_times):
        docs_recommended = doc_id_recommend[user_idx] # slate_size
        per_user[docs_recommended.numpy()-1] = 1.
    add_doc_recommend_times = tf.convert_to_tensor(add_doc_recommend_times, dtype=tf.float32)
    new_doc_recommend_times = tf.add(previous_state.get("doc_recommend_times"), add_doc_recommend_times)

    add_doc_click_times = np.zeros((self._num_users, self._num_docs))
    chosen_idx = user_response.get("choice")
    for user_idx, per_user in enumerate(add_doc_click_times):
      per_user_chosen_idx = chosen_idx[user_idx].numpy() #(1,)
      # User didn't choose any of the slate docs
      if(per_user_chosen_idx == slate_size):
        continue
      # Notice: the doc id starts from 1, however per_user index start from 0.
      docs_clicked = doc_id_recommend[user_idx][per_user_chosen_idx]
      per_user[docs_clicked.numpy()-1] = 1.
    add_doc_click_times = tf.convert_to_tensor(add_doc_click_times, dtype=tf.float32)
    new_doc_click_times = tf.add(previous_state.get("doc_click_times"), add_doc_click_times)

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
                       #Notice: each user needs a isolated space for their click and recommended history, so the shape is (num_users, num_docs)
        doc_recommend_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)), 
        doc_click_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)),)
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
