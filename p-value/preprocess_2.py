import torch
from tqdm import tqdm
# from dataset_tool import fast_merge
import numpy as np
from transformers import pipeline
import os
import pandas as pd
import json

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/Twibot-22'))
user = pd.read_json(os.path.join(base_path, 'newuser.json'))

user_text=list(user['description'])
tweet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'processed_data/id_tweet.json'))
with open(tweet_path, 'r') as f:
    each_user_tweets = json.load(f)

# feature_extract=pipeline('feature-extraction',model='roberta-base',tokenizer='roberta-base',device=1,padding=True, truncation=True,max_length=50, add_special_tokens = True)
feature_extract = pipeline(
    'feature-extraction',
    model='roberta-base',
    tokenizer='roberta-base',
    device=-1  # or 0
)

def Des_embbeding():
        print('Running feature1 embedding')
        path="processed_data/des_tensor.pt"
        if not os.path.exists(path):
            des_vec=[]
            for k,each in enumerate(tqdm(user_text)):
                if each is None:
                    des_vec.append(torch.zeros(768))
                else:
                    feature=torch.Tensor(feature_extract(each))
                    for (i,tensor) in enumerate(feature[0]):
                        if i==0:
                            feature_tensor=tensor
                        else:
                            feature_tensor+=tensor
                    feature_tensor/=feature.shape[1]
                    des_vec.append(feature_tensor)
                    
            des_tensor=torch.stack(des_vec,0)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(des_tensor,path)
        else:
            des_tensor=torch.load(path)
        print('Finished')
        return des_tensor

# def tweets_embedding():
#         print('Running feature2 embedding')
#         path="./processed_data/tweets_tensor.pt"
#         if True:
#             tweets_list=[]
#             for i in tqdm(range(len(each_user_tweets))):
#                 if len(each_user_tweets[str(i)])==0:
#                     total_each_person_tweets=torch.zeros(768)
#                 else:
#                     for j in range(len(each_user_tweets[str(i)])):
#                         each_tweet=each_user_tweets[str(i)][j]
#                         if each_tweet is None:
#                             total_word_tensor=torch.zeros(768)
#                         else:
#                             each_tweet_tensor=torch.tensor(feature_extract(each_tweet))
#                             for k,each_word_tensor in enumerate(each_tweet_tensor[0]):
#                                 if k==0:
#                                     total_word_tensor=each_word_tensor
#                                 else:
#                                     total_word_tensor+=each_word_tensor
#                             total_word_tensor/=each_tweet_tensor.shape[1]
#                         if j==0:
#                             total_each_person_tweets=total_word_tensor
#                         elif j==20:
#                             break
#                         else:
#                             total_each_person_tweets+=total_word_tensor
#                     if (j==20):
#                         total_each_person_tweets/=20
#                     else:
#                         total_each_person_tweets/=len(each_user_tweets[str(i)])
                        
#                 tweets_list.append(total_each_person_tweets)
                    
#             tweet_tensor=torch.stack(tweets_list)
#             torch.save(tweet_tensor,"./processed_data/tweets_tensor.pt")
                        
#         else:
#             tweets_tensor=torch.load(path)
#         print('Finished')


def tweets_embedding(each_user_tweets, feature_extract, max_tweets_per_user=20):
    print('Running feature2 embedding')
    path = "./processed_data/tweets_tensor.pt"

    tweets_list = []

    for i in tqdm(range(len(each_user_tweets))):
        user_tweets = each_user_tweets.get(str(i), [])

        if len(user_tweets) == 0:
            tweets_list.append(torch.zeros(768))
            continue

        total_each_person_tweets = torch.zeros(768)
        valid_tweet_count = 0

        for j, tweet in enumerate(user_tweets):
            if j >= max_tweets_per_user:
                break

            if tweet is None or not isinstance(tweet, str) or tweet.strip() == "":
                total_word_tensor = torch.zeros(768)
            else:
                try:
                    output = feature_extract(tweet, truncation=True, padding=True, max_length=50)
                    output_tensor = torch.tensor(output[0], dtype=torch.float32)  # shape: [seq_len, 768]

                    if output_tensor.ndim != 2 or output_tensor.shape[1] != 768:
                        raise ValueError(f"Unexpected shape: {output_tensor.shape}")

                    # Average pooling over tokens → [768]
                    total_word_tensor = output_tensor.mean(dim=0)

                except Exception as e:
                    print(f"[Warning] Error processing tweet {j} for user {i}: {e}")
                    total_word_tensor = torch.zeros(768)

            if valid_tweet_count == 0:
                total_each_person_tweets = total_word_tensor
            else:
                total_each_person_tweets += total_word_tensor

            valid_tweet_count += 1

        if valid_tweet_count > 0:
            total_each_person_tweets /= valid_tweet_count

        tweets_list.append(total_each_person_tweets)

    tweet_tensor = torch.stack(tweets_list)

    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    torch.save(tweet_tensor, path)
    print('Finished')
    
# Des_embbeding()
tweets_embedding(each_user_tweets, feature_extract)
