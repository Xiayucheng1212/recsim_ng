# coding=utf-8
# Copyright 2022 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Line as: python3
"""WIP: For testing differentiable interest evolution networks."""

from typing import Any, Callable, Collection, Sequence, Text, Optional

from recsim_ng.core import network as network_lib
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf

Network = network_lib.Network
Variable = variable.Variable


def reset_optimizer(learning_rate):
  return tf.keras.optimizers.SGD(learning_rate)


def distributed_train_step(
    tf_runtime,
    horizon,
    global_batch,
    trainable_variables,
    metric_to_optimize='reward',
    optimizer = None
):
  """Extracts gradient update and training variables for updating network."""
  with tf.GradientTape() as tape:
    last_state = tf_runtime.execute(num_steps=horizon)#TODO
    # --------------------------------
    # doc_recommend_times = network_lib.find_field(
    #     last_state, field_name='doc_recommend_times')
    # doc_recommend_times = doc_recommend_times.get("corpus state") + 0.0001
    # doc_click_times = network_lib.find_field(
    #     last_state, field_name='doc_click_times')
    # doc_click_times = doc_click_times.get('corpus state')
    
    rewards = network_lib.find_field(
      last_state, field_name='cumulative_reward')
    utility_reward = rewards.get("metrics state")
    success_reward = rewards.get("final metrics state")
    utility_reward = tf.reduce_mean(utility_reward)
    success_reward = tf.reduce_mean(success_reward)
    # --------------------------------
    last_metric_value = last_state['final metrics state'].get(metric_to_optimize)
    log_prob = last_state['slate docs_log_prob_accum'].get('doc_ranks')
    objective = -tf.tensordot(tf.stop_gradient(last_metric_value), log_prob, 1)
    objective /= float(global_batch)
  grads = tape.gradient(objective, trainable_variables)
  if optimizer:
    grads_and_vars = list(zip(grads, trainable_variables))
    optimizer.apply_gradients(grads_and_vars)
  return grads, objective, tf.reduce_mean(last_metric_value), success_reward, utility_reward


def make_runtime(variables):
  """Makes simulation + policy log-prob runtime."""
  variables = list(variables)
  slate_var = [var for var in variables if 'slate docs' == var.name]
  log_prob_var = log_probability.log_prob_variables_from_direct_output(
      slate_var)
  accumulator = log_probability.log_prob_accumulator_variables(log_prob_var)
  tf_runtime = runtime.TFRuntime(
      network=network_lib.Network(
          variables=list(variables) + list(log_prob_var) + list(accumulator)),
      graph_compile=False)
  return tf_runtime


def make_train_step(
    tf_runtime,
    horizon,
    global_batch,
    trainable_variables,
    metric_to_optimize,
    optimizer = None
):
  """Wraps a traced training step function for use in learning loops."""

  @tf.function
  def distributed_grad_and_train():
    return distributed_train_step(tf_runtime, horizon, global_batch,
                                  trainable_variables, metric_to_optimize,
                                  optimizer)

  return distributed_grad_and_train


def run_simulation(
    num_training_steps,
    horizon,
    global_batch,
    learning_rate,
    simulation_variables,
    trainable_variables,
    metric_to_optimize = 'reward',
):
  tf.config.run_functions_eagerly(True)
  """Runs simulation over multiple horizon steps while learning policy vars."""
  optimizer = reset_optimizer(learning_rate)
  tf_runtime = make_runtime(simulation_variables)
  train_step = make_train_step(tf_runtime, horizon, global_batch,
                               trainable_variables, metric_to_optimize,
                               optimizer)
  success_rate = 0.0
  utility = 0.0
  for i in range(num_training_steps):
    _ ,_ ,_ , now_success_rate, now_utility=train_step()
    success_rate+=now_success_rate
    utility+=now_utility
  print("success rate:", success_rate/(horizon*num_training_steps))
  return success_rate/(horizon*num_training_steps) , utility, utility/(num_training_steps * horizon)
