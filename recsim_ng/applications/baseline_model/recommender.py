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

"""Recommendation agents."""
import tensorflow.keras as keras
import random
import gin
from gym import spaces
import numpy as np
from recsim_ng.core import value
from recsim_ng.entities.choice_models import selectors as selector_lib
from recsim_ng.entities.choice_models import affinities as affinity_lib
from recsim_ng.entities.recommendation import recommender
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import estimation
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space

class GeneralizedLinearModel(tf.keras.Model):
    """A tf.keras model that returns one score for the input document. 
    Also, the weight in this linear model will be used as the user interest for the #slate_docs. """
    def __init__(self, num_docs, doc_embed_dim, num_users = 1):
        super().__init__(name="GeneralizedLinearModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._doc_embed_dim = doc_embed_dim
        self._net = tf.keras.Sequential(name="recs_linear")
        self._net.add(tf.keras.layers.Dense(1, input_shape=(self._doc_embed_dim,), activation='sigmoid'))

    def call(self, doc_feature):
        score = self._net(doc_feature)
        return score
    
@gin.configurable
class GeneralizedLinearRecommender(recommender.BaseRecommender):
   """A generalized linear based recommender implementation."""
   def __init__(self, config, model_ctor = GeneralizedLinearModel, name="Recommender_GLM"):
    super().__init__(config, name=name)
    self._num_docs = config.get("num_docs")
    self._num_topics = config.get("num_topics")
    self._doc_embed_dim = config.get("doc_embed_dim")
    self._epsilon = float(config.get("epsilon"))
    self._optimizer = keras.optimizers.SGD(0.1)
    self._model = model_ctor(self._num_docs, self._doc_embed_dim, self._num_users)
    self._train_acc = keras.metrics.BinaryAccuracy()
    self._train_loss = keras.metrics.Mean(name='train_loss')
    # The slate_size here is changed to num_docs, since we need to check all affinities in the available_docs
    self._affinity_model = affinity_lib.TargetPointSimilarity((self._num_users,), self._num_docs, 'negative_cosine')

   def initial_state(self):
     # Initializes the training loss and accuracy
     self._train_loss.reset_states()
     self._train_acc.reset_states()
     # Returns the model weights as a user interest
     # Notice: use trainable_weights instead of get_weights(), so that it can run inside the tensorflow graph.
     return Value(user_interest=tf.reshape(self._model.trainable_weights[0], [self._num_users, self._doc_embed_dim]))
   
   def next_state(self, previous_state, user_response, slate_docs):
     # Prepares training and ground_truth data
     del previous_state
     chosen_doc_idx = user_response.get("choice")
     training_y = tf.one_hot(chosen_doc_idx, depth = self._slate_size)
     # Upadte one step for each doc in the user_response
     # Although we only have 1 user, for the generality we still keep the for loop for num_users
     for i in range(self._num_users):
        training_x = slate_docs.as_dict["doc_features"][i]
        with tf.GradientTape() as tape:
            training_x_reshaped = tf.reshape(training_x, (self._slate_size, -1))
            training_y_reshaped = tf.reshape(training_y, (-1, 1))

            pred = self._model(training_x_reshaped)

            loss = keras.losses.binary_crossentropy(training_y_reshaped, pred)
            grads = tape.gradient(loss, self._model.trainable_variables) 
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))
     return Value(user_interest=tf.reshape(self._model.trainable_weights[0], [self._num_users, self._doc_embed_dim]))

   def slate_docs(self, previous_state, user_obs,
                 available_docs):
     # TODO: Implement KNN with Milvus
     del user_obs
     del previous_state
     explt_or_explr = random.uniform(0.0, 1.0)
     if explt_or_explr < self._epsilon: # Exploration Stage
        random_indices = tf.random.uniform(shape=[self._num_users, self._slate_size], minval=0, maxval=self._num_docs-1, dtype=tf.int32)
        slate = available_docs.map(lambda field: tf.gather(field, random_indices))
     else: # Exploitation Stage
        # The affinity model used -tf.keras.losses.cosine_similarity to calculate similarity
        # Ranging from (-1, 1), 1 means high similarity and -1 means high dissimilarity. 
        user_interest = tf.reshape(self._model.trainable_weights[0], [self._num_users, self._doc_embed_dim])
        affinities = self._affinity_model.affinities( 
            # user_interest: (num_users, n_features)
            user_interest,
            # doc_features: (slate_size, n_features)
            available_docs.get('doc_features')).get('affinities') + 2.0
        # Choose the top-k highest smilarity docs
        _, doc_indices = tf.math.top_k(affinities, k=self._slate_size)
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices))
     return slate
   
   # Returns the specs of the state and slate documents.
   def specs(self):
    # State_spec returns the specs of model's weights
    # Slate_docs_spec returns the specs of the recommended documents
    state_spec = ValueSpec(
      user_interest=Space(
            spaces.Box(
                low=np.ones(
                    (self._num_users, self._doc_embed_dim)) *
                -np.Inf,
                high=np.ones(
                    (self._num_users, self._doc_embed_dim)) *
                np.Inf))
    )
    slate_docs_spec = ValueSpec(
        doc_id=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_docs)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones((self._num_users, self._slate_size)) * self._num_topics)),
        doc_quality=Space(
            spaces.Box(
                low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
        doc_features=Space(
            spaces.Box(
                low=np.ones(
                    (self._num_users, self._slate_size, self._doc_embed_dim)) *
                -np.Inf,
                high=np.ones(
                    (self._num_users, self._slate_size, self._doc_embed_dim)) *
                np.Inf)))
    return state_spec.prefixed_with("state").union(
        slate_docs_spec.prefixed_with("slate"))