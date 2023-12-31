"""Demonsrate how we run the ecosystem simulation."""
import time

from absl import app
from recsim_ng.applications.low_cost_model import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

def run_simulation(num_runs, num_users, horizon, epsilon):
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
    sum_user_ctime = 0.0
    sum_ctr = 0.0
    for _ in range(num_runs):
        variables = simulation_config.create_one_user_glm_simulation_network(epsilon= epsilon, num_users=num_users, freeze_corpus=False)
        glm_network = network_lib.Network(variables=variables)
        with tf.compat.v1.Session().as_default():
            # @tf.function
            def run_one_simulation(network = glm_network):
                tf_runtime = runtime.TFRuntime(network=network)
                final_value = tf_runtime.execute(num_steps=horizon)
                # print("final_value:", final_value.get('metrics state'))
                rewards = network_lib.find_field(
                    final_value, field_name='cumulative_reward')
                success_reward = rewards.get("metrics state")
                single_run_reward = network_lib.find_field(
                    final_value, field_name='reward')
                ctr_reward = single_run_reward.get("final metrics state")
                
                
                print("final_reward:", ctr_reward[0])
                return success_reward[0] / horizon, ctr_reward[0]
        results = run_one_simulation()
        sum_ctr += results[1]
        sum_user_ctime += results[0]
    
    ctime_mean = sum_user_ctime / num_runs
    ctr_mean = sum_ctr / num_runs
    return ctime_mean, ctr_mean

def main(argv):
    del argv
    num_runs = 3
    num_users = 1
    horizon = 100
    epsilon = 0.0
    t_begin = time.time()
    reward_mean, avg_ctr = run_simulation(num_runs, num_users, horizon, epsilon)
    print('Elapsed time: %.3f seconds' %(time.time() - t_begin))
    print('Average reward: %f' %reward_mean)
    print('Average ctr: %f' %avg_ctr)

if __name__ == '__main__':
  app.run(main)

# Experiment result:
# horizon = 1 -> reward is 0.0
# horizon = 100 -> reward around 0.023
# horizon = 500 -> reward around 0.093
# horion = 800 -> reward around 0.24
# horizon = 1000 -> reward around 0.30
# horizon = 5000 -> reward around 0.58
# As long as we set the epsilon > 0.0, the reward will largely fluctuate.