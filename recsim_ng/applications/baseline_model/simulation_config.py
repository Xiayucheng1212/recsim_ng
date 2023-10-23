"""Configuration parameters for running ecosystem simulation."""
from typing import Collection
import gin
import numpy as np
from recsim_ng.applications.baseline_model import corpus
from recsim_ng.applications.baseline_model import metrics
from recsim_ng.applications.baseline_model import recommender
from recsim_ng.applications.baseline_model import user_interest
from recsim_ng.core import variable
from recsim_ng.stories import recommendation_simulation as simulation

Variable = variable.Variable

@gin.configurable
def create_cf_simulation_network(
    num_users = 1,
    num_topics = 42,
    slate_size = 6,
    doc_embed_dim= 2048,
    freeze_user = True,
    history_length = 15,
):
    num_docs = 9750 #9750
    config = {
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'history_length': history_length,
        'data_path': './str_embed/data'
    }
    if freeze_user:
        user_init = lambda config: user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init(config))
    else:
        user_ctor = user_interest.InterestEvolutionUser

    return simulation.recs_story(config, 
                    user_ctor,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.CollabFilteringRecommender,
                    metrics.SuccessRateMetrics,
                    metrics.ClickThroughRateAsRewardMetrics)

@gin.configurable
def create_random_simulation_network(
    num_users = 1,
    num_topics = 42,
    slate_size = 6,
    doc_embed_dim= 2048,
    freeze_user = True,
):
    num_docs = 9750 #9750
    config = {
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'data_path': './str_embed/data'
    }

    if freeze_user:
        user_init = lambda config: user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init(config))
    else:
        user_ctor = user_interest.InterestEvolutionUser

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
    slate_size=2,
    freeze_user = True,
    epsilon = 0.4,
):
    """Returns a network for the LinUCB simulation."""
    num_docs = 20
    config = {
        'alpha': alpha,
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'data_path': './str_embed/data',
        'epsilon': epsilon,
    }

    if freeze_user:
        user_init = lambda config: user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init(config))
    else:
        user_ctor = user_interest.InterestEvolutionUser

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
    history_length = 15,
    freeze_user = True
):
    num_docs = 9750
    config = {
        'epsilon': epsilon,
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'history_length': history_length,
        'data_path': './str_embed/data'
    }

    if freeze_user:
        user_init = lambda config: user_interest.InterestEvolutionUser(  # pylint: disable=g-long-lambda
            config).initial_state()
        user_ctor = lambda config: user_interest.StaticUser(config, user_init(config))
    else:
        user_ctor = user_interest.InterestEvolutionUser

    return simulation.recs_story(config, 
                    user_ctor,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.GeneralizedLinearRecommender,
                    metrics.SuccessRateMetrics,
                    metrics.ClickThroughRateAsRewardMetrics)

@gin.configurable
def create_one_user_glm_simulation_network(
    epsilon,
    num_users = 1,
    doc_embed_dim = 32,
    slate_size = 2,
    freeze_corpus = True
):
    """Retuns a network for the glm-based recommender simulation with viable corpus."""
    num_docs = 20
    config = {
        'epsilon': epsilon,
        'num_topics': doc_embed_dim,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
    }
    if freeze_corpus:
        corpus_init = lambda config: corpus.CorpusWithTopicAndQuality(  # pylint: disable=g-long-lambda
            config).initial_state()
        corpus_ctor = lambda config: corpus.StaticCorpus(config, corpus_init(config))
    else:
        corpus_ctor = corpus.CorpusWithTopicAndQuality

    return simulation.recs_story(config, 
                                 user_interest.InterestEvolutionUser,
                                 corpus_ctor,
                                 recommender.GeneralizedLinearRecommender,
                                 metrics.SuccessRateMetrics,
                                 metrics.ClickThroughRateAsRewardMetrics)