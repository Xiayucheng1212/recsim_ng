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
    def __init__(self, num_docs, doc_embed_dim, num_users = 5):
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
    self._optimizer = [keras.optimizers.SGD(0.1) for i in range(self._num_users)]
    self._model = [model_ctor(self._num_docs, self._doc_embed_dim, self._num_users) for i in range(self._num_users)]
    self._train_acc = keras.metrics.BinaryAccuracy()
    self._train_loss = keras.metrics.Mean(name='train_loss')
    # The slate_size here is changed to num_docs, since we need to check all affinities in the available_docs
    self._affinity_model = affinity_lib.TargetPointSimilarity((self._num_users,), self._num_docs, 'negative_cosine')
    self._iteration = 0
    self._last_phase = 1 # 1 means exploration, 0 means exploitation

   def initial_state(self):
     # Initializes the training loss and accuracy
     self._train_loss.reset_states()
     self._train_acc.reset_states()
     # Returns the model weights as a user interest
     # Notice: use trainable_weights instead of get_weights(), so that it can run inside the tensorflow graph.
     all_trainable_weights = [self._model[i].trainable_weights[0] for i in range(self._num_users)]
     return Value(user_interest=tf.reshape(all_trainable_weights, [self._num_users, self._doc_embed_dim]))
   
   def next_state(self, previous_state, user_response, slate_docs):
     # Prepares training and ground_truth data
     # Upadte one step for each doc in the user_response
     # Although we only have 1 user, for the generality we still keep the for loop for num_users
     if self._last_phase == 0: # only exploration phase can update the model
        return previous_state
     for i in range(self._num_users):
        chosen_doc_idx = user_response.get("choice")[i]
        current_slate_size = self._slate_size
        training_y = tf.one_hot(chosen_doc_idx, depth = self._slate_size)
        training_x = slate_docs.as_dict["doc_features"][i]
        # Add Oversampling data
        if chosen_doc_idx != self._slate_size:
            oversample_num = int(self._slate_size/3) # half of the slate_size is clicked docs
            oversample_y = tf.ones((oversample_num))
            training_y = tf.concat([training_y, oversample_y], axis=0)
            oversample_x = tf.reshape(tf.repeat(training_x[chosen_doc_idx], repeats = oversample_num, axis=0), (-1, self._doc_embed_dim))
            training_x = tf.concat([training_x, oversample_x], axis=0)
            current_slate_size = self._slate_size + oversample_num
            #-------------------------------------------------------------------------
            training_x = tf.reshape(training_x[chosen_doc_idx],(-1, self._doc_embed_dim))
            training_y = tf.ones((1))
            current_slate_size = 1
            #-------------------------------------------------------------------------

        with tf.GradientTape() as tape:
            training_x_reshaped = tf.reshape(training_x, (current_slate_size, -1))
            training_y_reshaped = tf.reshape(training_y, (current_slate_size))
            
            pred = tf.reshape(self._model[i](training_x_reshaped), (current_slate_size))
            loss = keras.losses.BinaryCrossentropy()(training_y_reshaped, pred)
        grads = tape.gradient(loss, self._model[i].trainable_variables) 
        self._optimizer[i].apply_gradients(zip(grads, self._model[i].trainable_variables))
        # if self._iteration % 200 == 0:
        #     print("loss:", loss)
     all_trainable_weights = [self._model[i].trainable_weights[0] for i in range(self._num_users)]

     return Value(user_interest=tf.reshape(all_trainable_weights, [self._num_users, self._doc_embed_dim]))

   def slate_docs(self, previous_state, user_obs,
                 available_docs):
     # TODO: Implement KNN with Milvus
     del user_obs
     del previous_state
     self._iteration += 1
     if self._epsilon > 0.0:
        if self._iteration%500 == 0:
           self._epsilon = self._epsilon * 0.9

     explt_or_explr = random.uniform(0.0, 1.0)
     if explt_or_explr < self._epsilon: # Exploration Stage
        self._last_phase = 1
        random_indices = tf.random.uniform(shape=[self._num_users, self._slate_size], minval=0, maxval=self._num_docs-1, dtype=tf.int32)
        slate = available_docs.map(lambda field: tf.gather(field, random_indices) if field.shape != [self._num_users, self._num_docs] else field )
     else: # Exploitation Stage
        # The affinity model used -tf.keras.losses.cosine_similarity to calculate similarity
        # Ranging from (-1, 1), 1 means high similarity and -1 means high dissimilarity. 
        self._last_phase = 0
        all_trainable_weights = [self._model[i].trainable_weights[0] for i in range(self._num_users)]
        user_interest = tf.reshape(all_trainable_weights, [self._num_users, self._doc_embed_dim])
        affinities = self._affinity_model.affinities( 
            # user_interest: (num_users, n_features)
            user_interest,
            # doc_features: (slate_size, n_features)
            available_docs.get('doc_features')).get('affinities') + 2.0
        # Choose the top-k highest smilarity docs
        _, doc_indices = tf.math.top_k(-1*affinities, k=self._slate_size)
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices) if field.shape != [self._num_users, self._num_docs] else field )
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
                np.Inf)),
        doc_vector=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
        doc_recommend_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)), 
        doc_click_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)),)
    return state_spec.prefixed_with("state").union(
        slate_docs_spec.prefixed_with("slate"))
    

@gin.configurable
class RandomRecommender(recommender.BaseRecommender):
   """A generalized linear based recommender implementation."""
   def __init__(self, config, name="Recommender_Random"):
    super().__init__(config, name=name)
    self._num_docs = config.get("num_docs")
    self._num_topics = config.get("num_topics")
    self._doc_embed_dim = config.get("doc_embed_dim")

   def initial_state(self):
    return Value(user_interest=tf.ones((self._num_users, self._doc_embed_dim)))
   
   def next_state(self, previous_state, user_response, slate_docs):
    del previous_state, user_response, slate_docs
    return Value(user_interest=tf.ones((self._num_users, self._doc_embed_dim)))
    
   def slate_docs(self, previous_state, user_obs,
                 available_docs):
     # TODO: Implement KNN with Milvus
        del user_obs
        del previous_state
        random_indices = tf.random.uniform(shape=[self._num_users, self._slate_size], minval=0, maxval=self._num_docs-1, dtype=tf.int32)
        slate = available_docs.map(lambda field: tf.gather(field, random_indices) if field.shape != [self._num_users, self._num_docs] else field )
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
                np.Inf)),
        doc_vector=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
        doc_recommend_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)), 
        doc_click_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)),)
    return state_spec.prefixed_with("state").union(
        slate_docs_spec.prefixed_with("slate"))
    
    
    
class CollabFilteringModel(tf.keras.Model):
  """A tf.keras model that returns score for each (user, document) pair."""

  def __init__(self, num_users, num_docs, doc_embed_dim,
               history_length):
    super().__init__(name="CollabFilteringModel")
    self._num_users = num_users
    self._history_length = history_length
    self._num_docs = num_docs
    self._doc_embed_dim = doc_embed_dim
    self._doc_proposal_embeddings = tf.keras.layers.Embedding(
        num_docs + 1,
        doc_embed_dim,
        embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
        mask_zero=True,
        name="doc_prop_embedding")
    self._doc_embeddings = tf.keras.layers.Embedding(
        num_docs + 1,
        doc_embed_dim,
        embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
        mask_zero=True,
        name="doc_embedding")
    self._net = tf.keras.Sequential(name="recs")
    self._net.add(tf.keras.layers.Dense(32))
    self._net.add(tf.keras.layers.LeakyReLU())
    self._net.add(
        tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

  def call(self, doc_id_history, doc_features):
    # Map doc id to embedding.
    # [num_users, history_length, embed_dim]
    doc_history_embeddings = self._doc_embeddings(doc_id_history)
    
    # Flatten and run through network to encode history.
    user_features = tf.reshape(doc_history_embeddings, (self._num_users, -1))
    #user_embeddings shape : (num_users, doc_embed_dim)
    user_embeddings = self._net(user_features)
    # Score is an inner product between the proposal embeddings(changed to contextual from corpus) and the encoded
    # history.
    #doc_features shape: (num_docs, doc_embed_dim)
    scores = tf.einsum("ik, jk->ij", user_embeddings, doc_features)
    #scores shape : (num_users, num_docs)
    return scores


@gin.configurable
class CollabFilteringRecommender(recommender.BaseRecommender):
  """A collaborative filtering based recommender implementation."""

  def __init__(self,
               config,
               model_ctor = CollabFilteringModel,
               name="Recommender"):  # pytype: disable=annotation-type-mismatch  # typed-keras
    super().__init__(config, name=name)
    self._history_length = config["history_length"]
    self._num_docs = config.get("num_docs")
    self._num_topics = config.get("num_topics")
    self._doc_embed_dim = 110#config.get("doc_embed_dim")
    self._model = model_ctor(self._num_users, self._num_docs, self._doc_embed_dim,
                             self._history_length)
    doc_history_model = estimation.FiniteHistoryStateModel(
        history_length=self._history_length,
        observation_shape=(),
        batch_shape=(self._num_users,),
        dtype=tf.int32)
    self._doc_history = dynamic.NoOPOrContinueStateModel(
        doc_history_model, batch_ndims=1)
    
    self._document_sampler = selector_lib.IteratedMultinomialLogitChoiceModel(
        self._slate_size, (self._num_users,),
        -np.Inf * tf.ones(self._num_users))

  def initial_state(self):
    """The initial state value."""
    doc_history_initial = self._doc_history.initial_state().prefixed_with(
        "doc_history")
    return doc_history_initial

  def next_state(self, previous_state, user_response,
                 slate_docs):
    """The state value after the initial value."""
    chosen_doc_idx = user_response.get("choice")
    chosen_doc_features = selector_lib.get_chosen(slate_docs, chosen_doc_idx)
    # Update doc_id history.
    doc_consumed = tf.reshape(
        chosen_doc_features.get("doc_id"), [self._num_users])
    # We update histories of only users who chose a doc.
    no_choice = tf.equal(user_response.get("choice"),
                         self._slate_size)[Ellipsis, tf.newaxis]
    next_doc_id_history = self._doc_history.next_state(
        previous_state.get("doc_history"),
        Value(input=doc_consumed,
              condition=no_choice)).prefixed_with("doc_history")
    return next_doc_id_history

  def slate_docs(self, previous_state, user_obs,
                 available_docs):
    """The slate_docs value."""
    del user_obs
    docid_history = previous_state.get("doc_history").get("state")
    # pass the contextual doc embeddings to the model
    scores = self._model(docid_history, available_docs.get("doc_features"))
    doc_indices = self._document_sampler.choice(scores).get("choice")
    slate = available_docs.map(lambda field: tf.gather(field, doc_indices) if field.shape != [self._num_users, self._num_docs] else field)
    return slate.union(Value(doc_ranks=doc_indices))

  def specs(self):
    state_spec = self._doc_history.specs().prefixed_with("doc_history")
    slate_docs_spec = ValueSpec(
        doc_ranks=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones(
                    (self._num_users, self._num_docs)) * self._num_docs)),
        doc_id=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_docs)),
        doc_topic=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._slate_size)),
                high=np.ones(
                    (self._num_users, self._slate_size)) * self._num_topics)),
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
                np.Inf)),
        doc_vector=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
        doc_recommend_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)), 
        doc_click_times=Space(
            spaces.Box(
                low=np.zeros((self._num_users, self._num_docs)),
                high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)))
    return state_spec.prefixed_with("state").union(
        slate_docs_spec.prefixed_with("slate"))

@gin.configurable
class LinearUCBRecommender(recommender.BaseRecommender):
    """A linear UCB based recommender implementation. See https://zhuanlan.zhihu.com/p/545790329 for more details."""
    def __init__(self, config,alpha=0.5, epsilon=0.3, name="LinearUCB"):
      super().__init__(config, name=name)
      self._num_docs = config.get("num_docs")
      self._num_topics = config.get("num_topics")
      self._doc_embed_dim = 110#config.get("doc_embed_dim")
      self._alpha = alpha
    #   self._epsilon = epsilon
      self._A = []
      self._invA = []
      self._b = []
      self._last_phase = 1 # 1 means exploration, 0 means exploitation

    def initial_state(self):
        """Parameter Initialization"""
        self._A = []
        self._invA = []
        self._b = []
        for u in range(self._num_users):
            A_per_user = []
            invA_per_user = []
            b_per_user = []
            for i in range(self._num_docs):
                A_per_user.append(np.eye(self._doc_embed_dim))
                invA_per_user.append(np.eye(self._doc_embed_dim))
                b_per_user.append(np.zeros((self._doc_embed_dim, 1)))
            self._A.append(A_per_user)
            self._invA.append(invA_per_user)
            self._b.append(b_per_user)
        # self._A shape: (num_users, num_docs, doc_embed_dim, doc_embed_dim)
        self._A = tf.convert_to_tensor(self._A, dtype=tf.float32)
        self._invA = tf.convert_to_tensor(self._invA, dtype=tf.float32)
        self._b = tf.convert_to_tensor(self._b, dtype=tf.float32)
        return Value(arm_param = self._A, arm_bias = self._b)

                
    def next_state(self, previous_state, user_response, slate_docs):
        """The state value after the initial value."""
        if self._last_phase == 0: # only exploration phase can update the model
            return previous_state        
        chosen_doc_idx = user_response.get("choice").numpy()
        # chosen_doc_idx shape: (num_users, 1)
        chosen_real_id = []
        for i in range(self._num_users):
            # User choose one document from the slate_docs
            if chosen_doc_idx[i] != self._slate_size:
               # chosen_real_id = slate_docs[chosen_doc_idx] - 1
               chosen_real_id.append([slate_docs.get("doc_id").numpy()[i][chosen_doc_idx[i]] - 1])
            # User choose none of the documents from the slate_docs
            else:
               chosen_real_id.append(slate_docs.get("doc_id").numpy()[i] - 1)
        # chosen real id start from 0, so we need to -1 from the slate_doc's doc_id
        for u in range(self._num_users):
            # we assume reward r = 1.0 as clicked and r = -1.0 as not clicked
            reward = 1.0 if chosen_doc_idx[u] != self._slate_size else -1.0
            chosen_doc_idx_per_user = chosen_doc_idx[u] if chosen_doc_idx[u] != self._slate_size else np.arange(self._slate_size)
            chosen_doc_idx_per_user = np.reshape(chosen_doc_idx_per_user, (1, -1)) #(one user, slate_size) or (one user, 1)
            # slate_docs_per_user shape: (1, slate_size, doc_embed_dim)
            slate_docs_per_user = Value(
               doc_features = tf.reshape(slate_docs.get("doc_features")[u], [1, -1, self._doc_embed_dim]))   
            # chosen_doc_features shape: (slate_size or 1, 1, doc_embed_dim)
            chosen_doc_features_per_user = tf.reshape(selector_lib.get_chosen(slate_docs_per_user, chosen_doc_idx_per_user).get("doc_features"), [-1, 1, self._doc_embed_dim])
            # Chosen_doc_features_per_user <-> chosen_real_id_per_user is one-to-one mapping
            for i in range(len(chosen_real_id[u])):
                A_update = tf.matmul(chosen_doc_features_per_user[i], chosen_doc_features_per_user[i], transpose_a=True)
                self._A = tf.tensor_scatter_nd_add(self._A,  indices=[[u, chosen_real_id[u][i]]], updates=[A_update])
                # Update formula of b = b + r * x
                b_update = tf.reshape(chosen_doc_features_per_user[i], (self._doc_embed_dim, 1)) * reward
                self._b = tf.tensor_scatter_nd_add(self._b, indices=[[u, chosen_real_id[u][i]]], updates=[b_update])
                invA_update = tf.linalg.inv(self._A[u][chosen_real_id[u][i]])
                self._invA = tf.tensor_scatter_nd_update(self._invA, indices=[[u, chosen_real_id[u][i]]], updates=[invA_update])

        return Value(arm_param = self._A, arm_bias = self._b)

    def slate_docs(self, previous_state, user_obs, available_docs):
        """The slate_docs value."""
        del user_obs
        del previous_state
        # doc_features shape: (num_docs, 1, doc_embed_dim)
        doc_features = tf.reshape(available_docs.get("doc_features"), [self._num_docs, 1, self._doc_embed_dim])
        # doc_ucb_scores shape: (num_users, num_docs)
        doc_ucb_scores = []
        
        self._last_phase = 0
        for u in range(self._num_users):
            doc_ucb_scores_per_user = []
            for i in range(self._num_docs):
                theta = tf.matmul(self._invA[u][i], self._b[u][i])
                score = tf.matmul(doc_features[i], theta) + self._alpha * tf.sqrt(tf.matmul(tf.matmul(doc_features[i], self._invA[u][i]), doc_features[i], transpose_b=True))
                doc_ucb_scores_per_user.append(score[0])

            doc_ucb_scores.append(doc_ucb_scores_per_user)
        doc_ucb_scores = tf.reshape(tf.convert_to_tensor(doc_ucb_scores, dtype=tf.float32), [self._num_users, self._num_docs])
        # Choose the top-k highest smilarity docs
        # doc_indices shape: (num_users, slate_size)
        _, doc_indices = tf.math.top_k(doc_ucb_scores, k=self._slate_size)
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices) if field.shape != [self._num_users, self._num_docs] else field )
        return slate

    def specs(self):
        state_spec = ValueSpec(
           arm_param = Space(
              spaces.Box(
                 low = np.ones((self._num_users, self._num_docs, self._doc_embed_dim, self._doc_embed_dim)) * -np.Inf,
                 high = np.ones((self._num_users, self._num_docs, self._doc_embed_dim, self._doc_embed_dim)) * np.Inf)),
           arm_bias = Space(
                spaces.Box(
                    low = np.ones((self._num_users, self._num_docs, self._doc_embed_dim, 1)) * -np.Inf,
                    high = np.ones((self._num_users, self._num_docs, self._doc_embed_dim, 1)) * np.Inf
                )
           )
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
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_topics)),
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
                    np.Inf)),
            doc_vector=Space(
            spaces.Box(
                low=np.zeros((self._num_docs, self._num_topics)),
                high=np.ones((self._num_docs, self._num_topics)))),
            doc_recommend_times=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._num_docs)),
                    high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)), 
            doc_click_times=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._num_docs)),
                    high=np.ones((self._num_users, self._num_docs)) * np.Inf, dtype=np.int32)))
        return state_spec.prefixed_with("state").union(slate_docs_spec.prefixed_with("slate"))