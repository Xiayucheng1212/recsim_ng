"""Configuration parameters for running ecosystem simulation."""
import functools
import tensorflow as tf
from typing import Collection
import gin
import numpy as np
from recsim_ng.applications.baseline_model import corpus
from recsim_ng.applications.baseline_model import metrics
from recsim_ng.applications.baseline_model import recommender
from recsim_ng.applications.baseline_model import user_interest
from recsim_ng.core import variable
from recsim_ng.stories import recommendation_simulation as simulation
from recsim_ng.lib.tensorflow import entity

Variable = variable.Variable

@gin.configurable
def create_cf_simulation_network(
    num_users = 1,
    num_topics = 42,
    slate_size = 6,
    doc_embed_dim= 2048,
    freeze_user = True,
    history_length = 15,
    more_interested_topics = None
):
  tf.config.run_functions_eagerly(True)
  """Returns a network for interests evolution simulation."""
  num_docs = 9750 #9750
  config = {
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'data_path': './str_embed/data',
        # History length for user representation in recommender.
        'history_length': history_length,
        'more_interested_topics': more_interested_topics
  }

  if  more_interested_topics != None:
      initial_interest_generator = user_interest.UserWithInterestedTopics(config)
  else: 
      initial_interest_generator = None

  if freeze_user:
      user_init = user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
          config, initial_interest_generator=initial_interest_generator).initial_state()
      user_ctor = lambda config: user_interest.StaticUser(config, user_init, interest_generator=initial_interest_generator)
  else:
      user_ctor = lambda config: user_interest.InterestEvolutionUser(config,initial_interest_generator=initial_interest_generator, freeze_user=freeze_user)
  var_fn = lambda: simulation.recs_story(  # pylint: disable=g-long-lambda
      config, user_ctor, corpus.CorpusWithEmbeddingsAndTopics,
      functools.partial(recommender.CollabFilteringRecommender),\
      metrics.SuccessRateMetrics,
      metrics.ClickThroughRateAsRewardMetrics)
  simulation_vars, trainable_vars = entity.story_with_trainable_variables(
      var_fn)
  return simulation_vars, trainable_vars['Recommender']


@gin.configurable
def create_random_simulation_network(
    num_users = 1,
    num_topics = 42,
    slate_size = 6,
    doc_embed_dim= 2048,
    freeze_user = True,
    more_interested_topics = None
):
    num_docs = 9750 #9750
    config = {
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'data_path': './str_embed/data',
        'more_interested_topics': more_interested_topics
    }
    if  more_interested_topics != None:
        initial_interest_generator = user_interest.UserWithInterestedTopics(config)
    else: 
        initial_interest_generator = None

    if freeze_user:
        user_init = user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config, initial_interest_generator=initial_interest_generator).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init, interest_generator=initial_interest_generator)
    else:
        user_ctor = lambda config: user_interest.InterestEvolutionUser(config, initial_interest_generator=initial_interest_generator, freeze_user=freeze_user)

    return simulation.recs_story(config, 
                    user_ctor,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.RandomRecommender,
                    metrics.SuccessRateMetrics,
                    metrics.ClickThroughRateAsRewardMetrics)

@gin.configurable
def create_linUCB_simulation_network(
    alpha,
    num_users=5,
    num_topics=42,
    doc_embed_dim=2048,
    slate_size=6,
    freeze_user = True,
    epsilon = 0.4,
    more_interested_topics = None
):
    """Returns a network for the LinUCB simulation."""
    num_docs = 9750
    config = {
        'alpha': alpha,
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'data_path': './str_embed/data',
        'epsilon': epsilon,
        'more_interested_topics': more_interested_topics
    }

    if  more_interested_topics != None:
        initial_interest_generator = user_interest.UserWithInterestedTopics(config)
    else: 
        initial_interest_generator = None

    if freeze_user:
        user_init = user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config, initial_interest_generator=initial_interest_generator).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init, interest_generator=initial_interest_generator)
    else:
        # dynamic user can only have random inital interest
        # TODO: implement inital interest for dynamic user
        user_ctor = lambda config: user_interest.InterestEvolutionUser(config, initial_interest_generator=initial_interest_generator, freeze_user=freeze_user)

    return simulation.recs_story(config, 
                    user_ctor,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.LinearUCBRecommender,
                    metrics.SuccessRateMetrics,
                    metrics.ClickThroughRateAsRewardMetrics)

@gin.configurable
def create_glm_contextual_simulation_network(
    epsilon,
    num_users=1,
    num_topics = 42,
    doc_embed_dim=2048,
    slate_size = 6, # 10 -> 0.1 avg ctr
    freeze_user = True,
    more_interested_topics = None
):
    num_docs = 9750
    config = {
        'epsilon': epsilon,
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'data_path': './str_embed/data',
        'more_interested_topics': more_interested_topics
    }
    if  more_interested_topics != None:
        initial_interest_generator = user_interest.UserWithInterestedTopics(config)
    else: 
        initial_interest_generator = None

    if freeze_user:
        user_init = user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config, initial_interest_generator=initial_interest_generator).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init, interest_generator=initial_interest_generator)
    else:
        user_ctor = lambda config: user_interest.InterestEvolutionUser(config, initial_interest_generator=initial_interest_generator, freeze_user=freeze_user)

    return simulation.recs_story(config, 
                    user_ctor,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.GeneralizedLinearRecommender,
                    metrics.SuccessRateMetrics,
                    metrics.ClickThroughRateAsRewardMetrics)