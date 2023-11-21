"""Demonsrate how we run the baseline model with embeddings."""
import time

from absl import app
from recsim_ng.applications.low_cost_model import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

def run_simulation(num_runs, num_users, more_interested_topics ,horizon, epsilon):
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
        variables = simulation_config.create_glm_contextual_simulation_network(epsilon= epsilon, num_users=num_users, more_interested_topics = more_interested_topics, freeze_user=False)
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
                success_reward = tf.reduce_mean(success_reward)
                ctr_reward = tf.reduce_mean(ctr_reward)
                print("final_reward:", ctr_reward)
                return success_reward / horizon, ctr_reward
        results = run_one_simulation()
        sum_ctr += results[1]
        sum_user_ctime += results[0]
    
    ctime_mean = sum_user_ctime / num_runs
    ctr_mean = sum_ctr / num_runs
    return ctime_mean, ctr_mean

def run_exp(horizon, more_interested_topics = False, epsilon = 0.8, f = None):
    num_runs = 3
    num_users = 3
    t_begin = time.time()
    reward_mean, avg_ctr = run_simulation(num_runs, num_users, more_interested_topics, horizon, epsilon = epsilon)
    print('Horizon: %d' %horizon)
    print(more_interested_topics)
    print('epsilon: %.2f' %epsilon)
    print('Elapsed time: %.3f seconds' %(time.time() - t_begin))
    print('Average reward: %f' %reward_mean)
    print('Average ctr: %f' %avg_ctr)

    # write to file
    f.write('Horizon: %d\n' %horizon)
    f.write('epsilon: %.2f\n' %epsilon)
    f.write('Elapsed time: %.3f seconds\n' %(time.time() - t_begin))
    f.write('Average reward: %f\n' %reward_mean)
    f.write('Average ctr: %f\n' %avg_ctr)


def main(argv):
    del argv
    horizons = [100,300,500, 700, 900, 1100, 1300]
    Epsilon1 = [0.4,0.4,0.4, 0.5, 0.5, 0.5, 0.5] # 1300 test 0.5
    Epsilon2 = [0.35,0.35,0.35, 0.3, 0.4, 0.4, 0.5]
    Epsilon3 = [0.45,0.45,0.45, 0.45, 0.5, 0.5, 0.5] # 1300 test 0.5
    ep_id = 0
    # open file
    f = open("./recsim_ng/expr_result/baseline_dynamic_exp.txt", "w")
    f.write("more_interested_topics = None\n")
    print("more_interested_topics = None")
    for i in horizons:
        run_exp(i,None,epsilon =Epsilon1[ep_id], f=f)
        ep_id+=1
    ep_id = 0
    f.write("more_interested_topics = False\n")
    print("more_interested_topics = False")
    for i in horizons:
        run_exp(i,False,epsilon =Epsilon2[ep_id], f=f)
        ep_id+=1
    ep_id = 0
    f.write("more_interested_topics = True\n")
    print("more_interested_topics = True")
    for i in horizons:
        run_exp(i,True,epsilon =Epsilon3[ep_id], f=f)
        ep_id+=1
    f.close()

if __name__ == '__main__':
  app.run(main)