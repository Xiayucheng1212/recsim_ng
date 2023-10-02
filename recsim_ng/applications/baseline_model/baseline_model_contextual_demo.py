"""Demonsrate how we run the baseline model with embeddings."""
import time

from absl import app
from recsim_ng.applications.baseline_model import baseline_model

def main(argv):
    del argv
    num_runs = 3
    num_users = 1
    horizon = 5000
    epsilon = 0.0
    t_begin = time.time()
    reward_mean = baseline_model.run_simulation(num_runs, num_users, horizon, epsilon)
    print('Elapsed time: %.3f seconds' %(time.time() - t_begin))
    print('Average reward: %f' %reward_mean)

if __name__ == '__main__':
  app.run(main)