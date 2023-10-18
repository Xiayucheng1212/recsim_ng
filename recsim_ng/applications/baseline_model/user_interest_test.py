import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.baseline_model import user_interest as ie_user
from recsim_ng.core import value
from recsim_ng.entities.choice_models import selectors as selector_lib
import tensorflow as tf
Value = value.Value

"""Test Baseline model for one user case"""
class BaselineInterestEvolutionUserTest(tf.test.TestCase):
    def setUp(self):
        super(BaselineInterestEvolutionUserTest, self).setUp()
        self._num_users = 1
        # Notice: set the _num_topics to document feature vector length
        self._num_topics = 5
        self._doc_embed_dim = self._num_topics 
        self._slate_size = 1
        self._no_click_mass = -np.Inf
        self._config = {
            'num_users': self._num_users,
            'num_topics': self._num_topics,
            'doc_embed_dim': self._doc_embed_dim,
            'slate_size': self._slate_size,
        }
    def test_initial_state(self):
        self._user = ie_user.InterestEvolutionUser(self._config, no_click_mass=self._no_click_mass)
        init_state = self._user.initial_state()
        user_interests = init_state.get('interest').get('state')
        print(user_interests)
        np.testing.assert_array_equal(
            [self._config['num_users'], self._config['doc_embed_dim']],
            np.shape(user_interests))


    def test_response(self):
        self._user = ie_user.InterestEvolutionUser(self._config, no_click_mass=self._no_click_mass)
        # Create a slate with one document only.
        doc_features = [[[1., 0., 0., 0., 0., 0.]]]
        doc_vector = [[[1., 0., 0., 0., 0.]]]
        print("doc_feature:-----", tf.constant(doc_features).shape)
        slate_docs = Value(
            doc_id=ed.Deterministic(loc=tf.constant([[1]])),
            doc_topic=ed.Deterministic(loc=tf.constant([[0]])),
            doc_quality=ed.Deterministic(
                loc=tf.constant([[0.]])),
            doc_features=ed.Deterministic(loc=tf.constant(doc_features)),
            doc_vector=ed.Deterministic(loc=tf.constant(doc_vector)),
        )
        user_state = Value(
            state=ed.Deterministic(
                loc=[[.1, .1, .1, .1, .1,]])).prefixed_with('interest')
        response = self.evaluate(
            self._user.next_response(user_state, slate_docs).as_dict)
        self.assertAllClose(
            {
                # The no click probability set to -np.Inf for all users.
                # Users will click on the only document presented to them.
                'choice': [0],
                # 'consumed_time': [0.5],
            },
            response)
        
class StaticUser(tf.test.TestCase):
    def setUp(self):
        super(StaticUser, self).setUp()
        self._num_users = 1
        # Notice: set the _num_topics to document feature vector length
        self._num_topics = 5
        self._doc_embed_dim = 6
        self._slate_size = 1
        self._no_click_mass = -np.Inf
        self._config = {
            'num_users': self._num_users,
            'num_topics': self._num_topics,
            'doc_embed_dim': self._doc_embed_dim,
            'slate_size': self._slate_size,
        }

    def test_next_state(self):
        init_state = ie_user.InterestEvolutionUser(self._config, no_click_mass=self._no_click_mass).initial_state()
        self._user = ie_user.StaticUser(self._config, init_state)
        next_state = self._user.next_state(init_state, None, None)
        user_interests = next_state.get('interest').get('state')
        print(user_interests)

        np.testing.assert_array_equal(
            init_state.get('interest').get('state'),
            user_interests)

if __name__ == '__main__':
  tf.test.main()