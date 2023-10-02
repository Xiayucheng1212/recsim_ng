"""Tests for recsim_ng.applications.baseline_model."""
import os

from absl.testing import parameterized
import numpy as np
from recsim_ng.applications.baseline_model import baseline_model
from recsim_ng.applications.baseline_model import simulation_config
from recsim_ng.applications.baseline_model import user_dynamic_interest as user
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import util
import tensorflow as tf



class BaselineModelSimulationTest(tf.test.TestCase, parameterized.TestCase):

    def setUp(self):
        super(BaselineModelSimulationTest, self).setUp()
        # tf.random.set_seed(0)

    def test_cumulative_reward_run(self):
        mean = baseline_model.run_simulation(
            1, # num_runs
            1, # num_users
            500, # horizon
            0.3 # epsilon
        )
        print("reward_mean: ", mean)
        self.assertAllClose(0.21, mean, rtol=0.1)
    # TODO: check the training error is lowering
    # TODO: check the meanings of horizon 
    # TODO: implement Contextual Corpus 
    # TODO: implement Milvus KNN
if __name__ == '__main__':
  print("--------",tf.executing_eagerly())
  tf.test.main()