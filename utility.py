import numpy as np
import torch
import pandas as pd
from tqdm import tqdm


def data_process(dataset_path):
    # reading the data frame
    # df = pd.read_csv("/content/drive/My Drive/AGJCSV/combained_validation_processed.csv")

    df = pd.read_csv(dataset_path)
    df.isnull().sum(), df[df.duplicated()].shape
    # df["sentiment"] = df["sentiment"].astype('category').cat.codes.astype('int')

    print("Dropping null and duplicates")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print("-" * 60)

    # Reduce Target length

    sentiment_count_map = dict(df["sentiment"].value_counts())
    sentiment_reduce_map = dict(zip(df["sentiment"].values, df["sentiment"].values))

    sentiment_reduce_map["afraid"] = "fear"
    sentiment_reduce_map["angry"] = "anger"
    sentiment_reduce_map["angry-disgusted"] = "anger"
    sentiment_reduce_map["anxiety"] = "anxious"
    sentiment_reduce_map["depression"] = "depressed"
    sentiment_reduce_map["disguest"] = "disgust"
    sentiment_reduce_map["disgusting"] = "disgust"
    sentiment_reduce_map["exited"] = "excitement"
    sentiment_reduce_map["fearful"] = "fear"
    sentiment_reduce_map["fun"] = "funny"
    sentiment_reduce_map["furious"] = "anger"
    sentiment_reduce_map["guilt"] = "remorse"
    sentiment_reduce_map["happiness"] = "happy"
    sentiment_reduce_map["hateful"] = "hate"
    sentiment_reduce_map["joy"] = "happy"
    sentiment_reduce_map["pessimism"] = "negative"
    sentiment_reduce_map["sadness"] = "sad"
    sentiment_reduce_map["scared"] = "scary"
    sentiment_reduce_map["suprise"] = "surprise"
    sentiment_reduce_map["surprised"] = "surprise"
    sentiment_reduce_map["very_funny"] = "funny"
    sentiment_reduce_map["very_negative"] = "negative"
    sentiment_reduce_map["worried"] = "worry"
    sentiment_reduce_map["not_relevant"] = "not-relevant"

    print("Sentiment mapping")
    print("Total sentiment before mapping")
    print(df["sentiment"].nunique())

    # Merging the sentiment
    df['sentiment'] = df['sentiment'].map(sentiment_reduce_map)

    print("Total sentiment after mapping")
    print(df["sentiment"].nunique())

    # Setting to category type for encoding
    df["sentiment"] = df["sentiment"].astype("category")

    # Count maps
    count_map = df["sentiment"].value_counts().to_dict()
    count_normalize = dict(df["sentiment"].value_counts(normalize=True))

    def get_low_target(value, count_map, frame):
        """This is used to display the sentiment having less then specified value"""
        target_list = []
        for key in count_map:
            # print(count_map[key])
            if count_map[key] < value:
                print("{}: {}: {}".format(key, count_map[key], count_map[key] / frame.shape[0] * 100))
                target_list.append(key)
        return target_list, len(target_list)

    print("-" * 60)
    print("Sentiments below 5000 data points")
    target_list, len_target_list = get_low_target(value=5000, count_map=count_map, frame=df)
    print(len_target_list)

    # Dropping minimum number of target (as of now dropping 5 targets)
    target_list = ['not-relevant',
                   'others', 'nocode', 'none', 'https']

    print("-" * 80)
    print("Sentiment to be dropped list, due to poor number of datapoints")
    print(target_list)

    def drop_target(target_list, df):
        """This is used to drop the particular sentiment"""
        df_new = df.drop(df[df.sentiment.isin(target_list)].index)
        df_new.to_csv("df_new.csv")
        df_new_reduced = pd.read_csv("df_new.csv")
        return df_new_reduced

    # df_new_reduced is reduced target data frame
    df_new_reduced = drop_target(target_list, df)

    print("-" * 80)
    print("Total Number of reduced sentiment (number of targets)")
    print(df_new_reduced["sentiment"].nunique())

    # Encode mapping (categorical encoding of the target)
    sentiment_map = dict(zip(df_new_reduced['sentiment'], df_new_reduced['sentiment'].astype("category").cat.codes))
    sentiment_demap = dict(zip(df_new_reduced["sentiment"].astype("category").cat.codes, df_new_reduced["sentiment"]))
    df_new_reduced['sentiment'] = df_new_reduced['sentiment'].map(sentiment_map)

    print("-" * 80)
    print("df.shape: {}".format(df_new_reduced.shape))
    print("df.columns: {}".format(df_new_reduced.columns))

    return df_new_reduced, sentiment_map, sentiment_demap


def get_weight(df):
    """This will give weights to encounter imbalanced class problem"""

    # Getting number of data points for each class
    weight_count = df["ENCODE_CAT"].value_counts(sort=False)

    # Weight of class c is the size of largest class divided by the size of class c.
    weight = weight_count.values.max() / weight_count.values
    class_weight = torch.tensor(weight.astype(np.float32))
    return class_weight


def calculate_accuracy(big_idx, targets):
    """"This is used to calculate the accuracy during training and validation"""
    n_correct = (big_idx == targets).sum().item()
    # print("^"*120)
    # print("Inside Accuracy:")
    # print("big_idx: {}".format(big_idx))
    # print("targets: {}".format(targets))
    # print("(big_idx==targets).sum().item(): {}".format((big_idx==targets).sum().item()))
    # print("^"*120)
    return n_correct


def get_max_len_sentence(df):
    """This function is used to get maximum length of the sentence/text (can be used for MAX_LEN param)"""
    max_len = 0
    for num in tqdm(df.index):
        # print(num)
        string = df["TITLE"][num].split(" ")
        if len(string) > max_len:
            max_len = len(string)
    print(max_len)
    return max_len
