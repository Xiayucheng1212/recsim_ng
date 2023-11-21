# Baseline Model

## Architecture
There are several components in this RecSim-NG system. They are User, Corpus(Documents), Metrics and Recommender. 
- User: Defines the user state transition model and user response. 
- Corpus: Defines features of ducuments, for exmaple: document quality, document topic, document feature vector and so on. 
- Metrics: Defines the reward for each interaction.
- Recommender: Gives the slate docs for the user.

## User
We use the same User model as in the applications/recsyc_partially_observable_rl.py. The user has an interest feature, which will evolve during the interections based on a state-transition model. As for the user_response, this model will choose the most similar document as it's response. The most similar document here refers to the document with the highest affinity value(i.e. highest cosine similarty). 

## Corpus
Again, it's the same Corpus model from the applications/recsyc_partially_observable_rl.py. It randomly samples the documents' topic, length and quality. Also, it will encode the topic into one-hot as the documents' feature vectors. In the future, we need to change the document feature to a contexual feature instead of randomly chosen vectors. 

## Metrics
It also comes from the applications/recsyc_partially_observable_rl.py. Returns the consume time as the reward for one interaction between user and recommender.

## Recommender
We design a Generalized Linear Model based recommender, which only contains 1 layer of logistic regression. The model weight inside will be extracted as the user interest we predicted. We will use the user interest to find the top-k similar document feature vectors as a "slate_docs". The slate docs means a bunch of documents recommended to the user. After the recommender get the user_response, the response will be used as the training data, and the model will be updated for a single step. Therefore, we assumed that after several rounds, the recommender shoud learn the user interest, and gives documents closely related to that. 

# Things need to be done

## Import Contextual Documents
To fully utilize the ability of Embedding Models like CLIP and BERT, we need to implement contextual documents as the Corpus. For example, we now can use MSCOCO dataset, encoding them into vectors and finally importing them as our documents. 

## Use Milvus as the KNN search engine
Since now we use the affinity function provided by RecsimNG, which will go through all the available documents. To make this system more scalable, we need to use Milvus as our KNN search engine. There are 2 functions that needs to be handled.
1. Randomly give slate docs -> Exploration
2. Use the user interest to find the top-k similar documents as the slate docs -> Exploitation

# Run Baseline Model
The entry point is applications/baseline_model/baseline_model_demo.py. So we only need to do `python baseline_model_demo.py`.
