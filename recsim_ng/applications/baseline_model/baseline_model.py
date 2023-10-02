"""For measuring social welfare of an ecosystem."""
from typing import Tuple
import numpy as np
from recsim_ng.applications.baseline_model import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

def run_simulation(num_runs, num_users, horizon, epsilon):
    """Runs ecosystem simulation multiple times and measures social welfare.

    Args:
    num_runs: Number of simulation runs. Must be a multiple of num_replicas.
    num_users: Number of users in this ecosystem.
    horizon: Length of each user trajectory. The number of iteration inside one time simulation
    epsilon: The threshold decides whether this recommendation is a exploration or a exploitation.

    Returns:
    The mean and standard error of cumulative user utility.
    """
    sum_user_ctime = 0.0
    for _ in range(num_runs):
        variables = simulation_config.create_one_user_glm_simulation_network(epsilon= epsilon, num_users=num_users)
        glm_network = network_lib.Network(variables=variables)
        with tf.compat.v1.Session().as_default():
            # @tf.function
            def run_one_simulation(network = glm_network):
                tf_runtime = runtime.TFRuntime(network=network)
                final_value = tf_runtime.execute(num_steps=horizon)
                # print("final_value:", final_value.get('metrics state'))
                unsued, final_reward = network_lib.find_unique_field(
                    final_value, field_name='cumulative_reward')
                print("final_reward:", final_reward.numpy())
                r = final_reward[0] / horizon
                return r
        
        sum_user_ctime += run_one_simulation().numpy()
    
    ctime_mean = sum_user_ctime / num_runs
    return ctime_mean
                