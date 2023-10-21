"""Tests for recsim_ng.applications.baseline_model.corpus."""

import edward2 as ed  # type: ignore
import numpy as np
from recsim_ng.applications.baseline_model import corpus as static_corpus
from recsim_ng.core import value
import tensorflow as tf

Value = value.Value


class StaticCorpusTest(tf.test.TestCase):

  def setUp(self):
    super(StaticCorpusTest, self).setUp()
    self._num_users = 3
    self._num_docs = 20
    self._num_topics = 42
    self._doc_embed_dim = 2048
    self._data_path = './str_embed/data'
    self._config = {
        'num_topics': self._num_topics,
        'num_docs': self._num_docs,
        'num_users': self._num_users,
        'doc_embed_dim': self._doc_embed_dim,
        'data_path': self._data_path
    }
    self._corpus = static_corpus.CorpusWithEmbeddingsAndTopics(self._config)

  def test_init_state(self):
      init_state = self._corpus.initial_state()
      init_state_dict = self.evaluate(init_state.as_dict)
      doc_id = init_state_dict['doc_id']
      doc_topic = init_state_dict['doc_topic']
      doc_quality = init_state_dict['doc_quality']
      doc_features = init_state_dict['doc_features']
      doc_recommend_times = init_state_dict['doc_recommend_times']
      doc_click_times = init_state_dict['doc_click_times']
      doc_vector = init_state_dict['doc_vector']
      np.testing.assert_array_equal([self._config['num_docs']], np.shape(doc_id))
      np.testing.assert_array_equal([self._config['num_docs']],
                                    np.shape(doc_topic))
      np.testing.assert_array_equal([self._config['num_docs']],
                                    np.shape(doc_quality))
      np.testing.assert_array_equal(
          [self._config['num_docs'], self._config['doc_embed_dim']],
          np.shape(doc_features))
      np.testing.assert_array_equal([self._config['num_users'],self._config['num_docs']],
                                    np.shape(doc_recommend_times))
      np.testing.assert_array_equal([self._config['num_users'],self._config['num_docs']],
                                    np.shape(doc_click_times))
      np.testing.assert_array_equal([self._config['num_docs'], self._config['num_topics']], np.shape(doc_vector))
  def test_next_state(self):
    init_state = self._corpus.initial_state()
    init_state_dict = self.evaluate(init_state.as_dict)
    # Static corpus does not change its state on any user response.
    user_response = Value(
      choice=ed.Deterministic(
                loc=tf.constant([1, 1, 1], dtype=tf.int32)),)
  
    slate_docs = Value(
            doc_id=tf.constant([[2, 3], [1,2], [3,4]], dtype=tf.int32)
    )
    actual = self._corpus.next_state(init_state, user_response, slate_docs)
    expected = np.zeros((self._num_users, self._num_docs))
    expected[0][1:3] = [1,1]
    expected[1][0:2] = [1,1]
    expected[2][2:4] = [1,1]
    expected = tf.convert_to_tensor(expected)
    print(actual.get('doc_click_times'))
    print(actual.get('doc_recommend_times'))
    # self.assertEqual(actual, expected)

  def test_available_documents(self):
    corpus_state = self._corpus.initial_state()
    actual = self._corpus.available_documents(corpus_state)
    self.assertAllClose(
        self.evaluate(corpus_state.as_dict), self.evaluate(actual.as_dict))


if __name__ == '__main__':
  tf.test.main()
