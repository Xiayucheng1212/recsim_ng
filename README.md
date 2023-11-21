# Low-Cost Recommendation System
Our low-cost recommendation system is developed leveraging the capabilities of Recsim NG, a robust simulator capable of emulating the user interaction process, document preparation, recommendation process, and metrics for evaluating recommendation accuracy.

To provide more details, our project is essentially a fork of the GitHub repository Recsim NG. Within the recsim_ng/applications/low_cost_model folder, we have implemented the new low-cost recommendation model, referred to as low_cost_model, along with other competing models such as cf_model, linucb_model, and random_model.

## Results
The metric used to evaluate the system in this study is the average recommendation success rate. For each recommendation, users have the option to select one document from a slate (a set of recommended documents). A reward of one is assigned if the user selects a document; otherwise, the reward is zero. The user setup is meticulously designed and detailed in the recsim_ng/applications/user_interest.py file—further information can be found there. Currently, the user probability for choosing one of the documents is set at 0.4.

### Accuracy Results
The accuracy of our low-cost recommendation system is competitive with other popular recommendation systems, particularly LinUCB. However, due to the limited maximum number of users (3) in the experiments, Collaborative-Filtering recommender faces challenges in learning from other users. Consequently, this explains the suboptimal performance of the cf_model.

![](/docs/pics/dynamic_user_none.png)
![](/docs/pics/dynamic_user_less.png) 
![](/docs/pics/dynamic_user_more.png) 

### Consumption Results
It is evident that the low-cost recommender excels in both time and memory efficiency, surpassing other recommendation systems.

![](/docs/pics/time_consumption.png)
![](/docs/pics/memory_usage.png)


##  RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems

RecSim NG, a probabilistic platform for multi-agent recommender systems
simulation. RecSimNG is a scalable, modular, differentiable simulator
implemented in Edward2 and TensorFlow. It offers: a powerful, general
probabilistic programming language for agent-behavior specification; an
XLA-based vectorized execution model for running simulations on accelerated
hardware; and tools for probabilistic inference and latent-variable model
learning, backed by automatic differentiation and tracing. We describe RecSim NG
and illustrate how it can be used to create transparent, configurable,
end-to-end models of a recommender ecosystem. Specifically, we present a
collection of use cases that demonstrate how the functionality described above
can help both researchers and practitioners easily develop and train novel
algorithms for recommender systems. Please refer to
[Mladenov et al](https://arxiv.org/abs/2103.08057) for the
high-level design of RecSim NG. Please cite the paper if you use the code from
this repository in your work.

<a id='Disclaimer'></a>

## Disclaimer

This is not an officially supported Google product.

## Installation and Sample Usage

It is recommended to install RecSim NG using
(https://pypi.org/project/recsim_ng).

```shell
pip install recsim_ng
```

Here are some sample commands you could use for testing the installation:

```
git clone https://github.com/google-research/recsim_ng
cd recsim_ng/recsim_ng/applications/ecosystem_simulation
python ecosystem_simulation_demo.py
```

## Tutorials

To get started, please check out our Colab tutorials. In
[**RecSim NG: Basics**](https://colab.research.google.com/github/google-research/recsim_ng/blob/master/recsim_ng/colab/RecSim_NG_Basics.ipynb),
we introduce the RecSim NG model and corresponding modeling APIs and runtime
library. We then demonstrate how we define a simulation using **entities**,
**behaviors**, and **stories**. Finally, we illustrate differentiable
simulation including model learning and inference.

In [**RecSim NG: Dealing With Uncertainty**](https://colab.research.google.com/github/google-research/recsim_ng/blob/master/recsim_ng/colab/RecSim_NG_Dealing_With_Uncertainty.ipynb),
we explicitly address the stochastics of the Markov process captured by a DBN.
We demonstrate how to use Edward2 in RecSim NG and show how to use the
corresponding RecSim NG APIs for inference and learning tasks. Finally, we
showcase how the uncertainty APIs of RecSim NG can be used within a
recommender-system model-learning application.

## Documentation


Please refer to the [white paper](https://arxiv.org/abs/2103.08057)
for the high-level design.
