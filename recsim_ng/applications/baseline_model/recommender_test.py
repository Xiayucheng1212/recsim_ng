"""Tests for recsim_ng.applications.recsys_partially_observable_rl.recommender."""

import functools

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.baseline_model import recommender as glm_recommender
from recsim_ng.core import value
import tensorflow as tf
import operator

Value = value.Value


class GLMRecommenderTest(tf.test.TestCase):
    def setUp(self):
        super(GLMRecommenderTest, self).setUp()
        self._num_users = 1
        self._num_docs = 5
        self._num_topics = 10
        self._doc_embed_dim = 10
        self._slate_size = 2
        self._epsilon = 0.6
        self._config = {
            'num_users': self._num_users,
            'num_docs': self._num_docs,
            'num_topics': self._num_topics,
            'doc_embed_dim': self._doc_embed_dim,
            'slate_size': self._slate_size,
            'epsilon': self._epsilon
        }

    def test_default_initialization(self):
        """Tests initialized state shape is correct"""
        self._recommender = glm_recommender.GeneralizedLinearRecommender(
            self._config)
        init_state = self._recommender.initial_state()
        init_state_dict = self.evaluate(init_state.as_dict)
        np.testing.assert_equal(
            (self._num_users, self._num_topics),
            init_state_dict['user_interest'].shape)

    def test_next_states(self):
        """Tests next state with the GLM model"""
        self._recommender = glm_recommender.GeneralizedLinearRecommender(
            self._config)
        init_state = self._recommender.initial_state()
        # Creates a dummy user response.
        mock_user_response = Value(
            choice=ed.Deterministic(
                loc=tf.constant([1], dtype=tf.int32)),
            consumed_time=ed.Deterministic(
                loc=tf.constant([2.2], dtype=tf.float32)))
        # Creates a dummy slate_docs with doc_features
        mock_slate_docs = Value(
            doc_id=tf.constant([[2, 3]], dtype=tf.int32),
            doc_features=ed.Deterministic(loc=tf.constant(
                [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]]
            ))
        )
        next_state = self._recommender.next_state(
            init_state, mock_user_response, mock_slate_docs)
        np.testing.assert_array_compare(
            operator.__ne__,
            init_state.as_dict['user_interest'],
            self.evaluate(next_state.as_dict)['user_interest']
        )
    
    # TODO: test multiple steps of next_states
        
    def test_slate_docs(self):
        available_docs = Value(
            doc_id=ed.Deterministic(
                loc=tf.range(
                    start=1, limit=self._config['num_docs'] + 1, dtype=tf.int32)),
            doc_topic=ed.Deterministic(loc=tf.ones((self._num_docs,))),
            doc_quality=ed.Normal(
                loc=tf.zeros((self._config['num_docs'],)), scale=0.1),
            doc_features=ed.Deterministic(
                loc=tf.ones((self._num_docs, self._num_topics)) * 1.0 /
                self._num_topics),
        )

        self._recommender = glm_recommender.GeneralizedLinearRecommender(
            self._config)
        slate_docs = self.evaluate(self._recommender.slate_docs({}, {}, available_docs).as_dict)

        self.assertCountEqual(
            ['doc_id', 'doc_topic', 'doc_quality', 'doc_features'],
            slate_docs.keys())
        np.testing.assert_array_equal(
            [self._config['num_users'], self._config['slate_size']],
            np.shape(slate_docs['doc_id']))
        np.testing.assert_array_equal(
            [self._config['num_users'], self._config['slate_size']],
            np.shape(slate_docs['doc_topic']))
        np.testing.assert_array_equal(
            [self._config['num_users'], self._config['slate_size']],
            np.shape(slate_docs['doc_quality']))
        np.testing.assert_array_equal([
            self._config['num_users'], self._config['slate_size'],
            self._config['num_topics']
        ], np.shape(slate_docs['doc_features']))
        

if __name__ == '__main__':
    tf.test.main()
