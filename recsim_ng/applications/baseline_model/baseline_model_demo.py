"""Demonsrate how we run the ecosystem simulation."""
import time

from absl import app
from recsim_ng.applications.baseline_model import baseline_model
from recsim_ng.applications.baseline_model import user_dynamic_interest as user
from recsim_ng.lib.tensorflow import util

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

# Experiment result:
# horizon = 1 -> reward is 0.0
# horizon = 100 -> reward around 0.023
# horizon = 500 -> reward around 0.093
# horion = 800 -> reward around 0.24
# horizon = 1000 -> reward around 0.30
# horizon = 5000 -> reward around 0.58
# As long as we set the epsilon > 0.0, the reward will largely fluctuate.