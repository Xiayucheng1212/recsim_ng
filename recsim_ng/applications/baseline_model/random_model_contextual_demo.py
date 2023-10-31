"""Demonsrate how we run the baseline model with embeddings."""
import time

from absl import app
from recsim_ng.applications.baseline_model import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

def run_simulation(num_runs, num_users, horizon):
    tf.config.run_functions_eagerly(True)
    """Runs ecosystem simulation multiple times and measures social welfare.

    Args:
    num_runs: Number of simulation runs. Must be a multiple of num_replicas.
    num_users: Number of users in this ecosystem.
    horizon: Length of each user trajectory. The number of iteration inside one time simulation
    epsilon: The threshold decides whether this recommendation is a exploration or a exploitation.

    Returns:
    The mean and standard error of cumulative user utility.
    """
    sum_utility_horizon_mean = 0.0
    sum_successrate = 0.0
    sum_utility = 0.0
    for _ in range(num_runs):
        variables = simulation_config.create_random_simulation_network(num_users=num_users, more_interested_topics=True)
        glm_network = network_lib.Network(variables=variables)
        with tf.compat.v1.Session().as_default():
            # @tf.function
            def run_one_simulation(network = glm_network):
                tf_runtime = runtime.TFRuntime(network=network)
                final_value = tf_runtime.execute(num_steps=horizon)
                # print("final_value:", final_value.get('metrics state'))
                rewards = network_lib.find_field(
                    final_value, field_name='cumulative_reward')
                utility_reward = rewards.get("metrics state")
                success_reward = rewards.get("final metrics state")
                utility_reward = tf.reduce_mean(utility_reward)
                success_reward = tf.reduce_mean(success_reward)
                print("success rate:", success_reward/horizon)
                return utility_reward / horizon, utility_reward, success_reward/horizon
        results = run_one_simulation()
        sum_successrate += results[2]
        sum_utility_horizon_mean += results[0]
        sum_utility += results[1]
    
    success_rate = sum_successrate / num_runs
    utility_mean = sum_utility / num_runs
    utility_horizon_mean = sum_utility_horizon_mean / num_runs
    return success_rate, utility_mean, utility_horizon_mean

def main(argv):
    del argv
    num_runs = 3
    num_users = 3
    horizon = 100
    t_begin = time.time()
    reward_mean, cumulate_utility_mean, utility_mean = run_simulation(num_runs, num_users, horizon)
    print('Elapsed time: %.3f seconds' %(time.time() - t_begin))
    print('Average successrate: %f' %reward_mean)
    print('Average cumulate utility: %f' %cumulate_utility_mean)
    print('Average utility: %f' %utility_mean)
    # slate = 6, horizon = 5000, num_doc= 9750 -> 0.1622
    # slate = 6, horizon = 1000, num_doc= 9750 -> 0.160916

if __name__ == '__main__':
  app.run(main)