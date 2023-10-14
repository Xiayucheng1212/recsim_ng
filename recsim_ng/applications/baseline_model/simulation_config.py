"""Configuration parameters for running ecosystem simulation."""
from typing import Collection
import gin
import numpy as np
from recsim_ng.applications.baseline_model import corpus
from recsim_ng.applications.baseline_model import metrics
from recsim_ng.applications.baseline_model import recommender
from recsim_ng.applications.baseline_model import user_dynamic_interest
from recsim_ng.core import variable
from recsim_ng.stories import recommendation_simulation as simulation

Variable = variable.Variable

@gin.configurable
def create_linUCB_simulation_network(
    alpha,
    num_users=5,
    num_topics=42,
    doc_embed_dim=1536,
    slate_size=2,
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
    }
    return simulation.recs_story(config, 
                    user_dynamic_interest.InterestEvolutionUser,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.LinearUCBRecommender,
                    metrics.SuccessRateMetrics,
                    metrics.ClickThroughRateAsRewardMetrics)

@gin.configurable
def create_glm_contextual_simulation_network(
    epsilon,
    num_users=5,
    num_topics = 42,
    doc_embed_dim=1536,
    slate_size = 2,
    history_length = 15
):
    num_docs = 20
    config = {
        'epsilon': epsilon,
        'num_topics': num_topics,
        'doc_embed_dim': doc_embed_dim,
        'num_users': num_users,
        'num_docs': num_docs,
        'slate_size': slate_size,
        'history_length': history_length,
    }
    return simulation.recs_story(config, 
                    user_dynamic_interest.InterestEvolutionUser,
                    corpus.CorpusWithEmbeddingsAndTopics,
                    recommender.CollabFilteringRecommender,
                    metrics.ClickThroughRateAsRewardMetrics)

@gin.configurable
def create_one_user_glm_simulation_network(
    epsilon,
    num_users = 1,
    doc_embed_dim = 32,
    slate_size = 3,
    freeze_corpus = True
):
    """Retuns a network for the glm-based recommender simulation with viable corpus."""
    num_docs = 100
    config = {
        'epsilon': epsilon,
        'num_topics': doc_embed_dim,
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
                                 user_dynamic_interest.InterestEvolutionUser,
                                 corpus_ctor,
                                 recommender.GeneralizedLinearRecommender,
                                 metrics.ConsumedTimeAsRewardMetrics)