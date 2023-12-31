"""Demonsrate how we run the baseline model with embeddings."""
import time

from absl import app
from recsim_ng.applications.low_cost_model import simulation_config
from recsim_ng.applications.low_cost_model import cf_interest_evolution_simulation
from recsim_ng.core import network as network_lib
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

def run_exp(horizon = 1000, more_interested_topics = None):
    num_runs = 3
    num_training_steps = 20
    horizon = horizon/num_training_steps
    num_users = 3
    history_length = 15
    t_begin = time.time()
    # -----------------------------------------
    sum_user_creward = 0.0
    sum_ctr = 0.0
    for _ in range(num_runs):
        variables, trainable_variables = (
        simulation_config.create_cf_simulation_network(
            num_users=num_users, history_length=history_length,more_interested_topics = more_interested_topics))
        
        results = cf_interest_evolution_simulation.run_simulation(
        num_training_steps=num_training_steps,
        horizon=horizon,
        global_batch=num_users,
        learning_rate=1e-4,
        simulation_variables=variables,
        trainable_variables=trainable_variables,
        metric_to_optimize='cumulative_reward')

        sum_ctr += results[1]
        sum_user_creward += results[0]
    
    creward_mean = sum_user_creward / num_runs
    avg_ctr = sum_ctr / num_runs

    # -----------------------------------------
    print('horizon: %d' %horizon*num_training_steps)
    print('Elapsed time: %.3f seconds' %(time.time() - t_begin))
    print('Average reward: %f' %creward_mean)
    print('Average ctr: %f' %avg_ctr)
    # 0.47 -> new doc_vector but epsilon 1.0 -> 0.9
    # 0.45 -> new doc_vector but epsilon 0.5 -> 0.45
    # slate = 6 -> 0.16429 with ep = 0.8 horizon = 5000
    # slate = 6 -> with ep = 0.5 horizon = 1000 -> 0.1609
    # slate = 6 -> with ep = 0.5 horizon = 1000 -> 0.16303 with oversampling


def main(argv):
    del argv
    horizons = [100,500,1000,5000]
    print("more_interested_topics = None")
    for i in horizons:
        run_exp(i,None)
    print("more_interested_topics = False")
    for i in horizons:
        run_exp(i,False)
    print("more_interested_topics = True")
    for i in horizons:
        run_exp(i,True)

if __name__ == '__main__':
  app.run(main)