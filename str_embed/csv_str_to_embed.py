import pandas as pd
import tiktoken
import openai
from openai.embeddings_utils import get_embedding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
print("?")
openai.api_key = "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"
embedding_model1 = "text-embedding-ada-002"
embedding_model2 = "text-similarity-babbage-001"
embedding_encoding = "cl100k_base"
input_datapath = "./data/csv_rawdata.csv"
output_datapath = "./data/embeddings.csv"

df = pd.read_csv(input_datapath)
df['category_encoded'] = label_encoder.fit_transform(df['category'])
df["combined"] = (
    "Title: " + df.category.str.strip() + "; Content: " + df.headline.str.strip()
)
#create group by encode number
number_each_encode = 240
result_dfs = []
for encode, group in df.groupby('category_encoded'):
    sampled_group = group.head(number_each_encode+60).dropna()
    sampled_group = sampled_group.head(number_each_encode)
    result_dfs.append(sampled_group)
df = pd.concat(result_dfs)

#create embedding
df["embedding_cat_headline"] = df.combined.apply(lambda x: get_embedding(x, engine=embedding_model1))
df["embedding_description"] = df.short_description.apply(lambda x: get_embedding(x, engine=embedding_model2))
df = df.drop(["date","link","authors","category","headline","short_description","combined"], axis='columns')
df.to_csv(output_datapath)