import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import config
import pickle
import re
import os
from sklearn.metrics import confusion_matrix
from IPython.display import display
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd


def data_process(dataset_path):
    # reading the data frame
    # df = pd.read_csv("/content/drive/My Drive/AGJCSV/combained_validation_processed.csv")

    df = pd.read_csv(dataset_path, encoding="ISO-8859-1")  # encoding to avoid UnicodeDecodeError
    df.isnull().sum(), df[df.duplicated()].shape
    # df["sentiment"] = df["sentiment"].astype('category').cat.codes.astype('int')

    print("Dropping null and duplicates")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    print("-" * 60)

    # Reduce Target length

    sentiment_count_map = dict(df["sentiment"].value_counts())
    sentiment_reduce_map = dict(zip(df["sentiment"].values, df["sentiment"].values))

    # sentiment_reduce_map["afraid"] = "fear"
    # sentiment_reduce_map["angry"] = "anger"
    # sentiment_reduce_map["angry-disgusted"] = "anger"
    # sentiment_reduce_map["anxiety"] = "anxious"
    # sentiment_reduce_map["depression"] = "depressed"
    # sentiment_reduce_map["disguest"] = "disgust"
    # sentiment_reduce_map["disgusting"] = "disgust"
    # sentiment_reduce_map["exited"] = "excitement"
    # sentiment_reduce_map["fearful"] = "fear"
    # sentiment_reduce_map["fun"] = "funny"
    # sentiment_reduce_map["furious"] = "anger"
    # sentiment_reduce_map["guilt"] = "remorse"
    # sentiment_reduce_map["happiness"] = "happy"
    # sentiment_reduce_map["hateful"] = "hate"
    # sentiment_reduce_map["joy"] = "happy"
    # sentiment_reduce_map["pessimism"] = "negative"
    # sentiment_reduce_map["sadness"] = "sad"
    # sentiment_reduce_map["scared"] = "scary"
    # sentiment_reduce_map["suprise"] = "surprise"
    # sentiment_reduce_map["surprised"] = "surprise"
    # sentiment_reduce_map["very_funny"] = "funny"
    # sentiment_reduce_map["very_negative"] = "negative"
    # sentiment_reduce_map["worried"] = "worry"
    # sentiment_reduce_map["not_relevant"] = "not-relevant"

    # print("Total sentiment before mapping")
    # print("Total number of sentiments: {}".format(df["sentiment"].nunique()))
    # # Merging the sentiment set 1
    # df['sentiment'] = df['sentiment'].map(sentiment_reduce_map)
    # print("After sentiment mapping Set_1")
    # print("Total number of sentiments: {}".format(df["sentiment"].nunique()))

    # # New mapping
    # sentiment_reduce_map["awe"] = "carving"
    # sentiment_reduce_map["entertaining"] = "ridicule"
    # sentiment_reduce_map["criticize"] = "ridicule"
    # sentiment_reduce_map["sarcasm"] = "ridicule"
    # sentiment_reduce_map["shame"] = "disgust"
    # sentiment_reduce_map["enthusiasm"] = "surprise"
    # sentiment_reduce_map["relief"] = "happy"
    # sentiment_reduce_map["excitement"] = "happy"
    # sentiment_reduce_map["love"] = "happy"
    # sentiment_reduce_map["horror"] = "anxious"
    # sentiment_reduce_map["pain"] = "anxious"
    # sentiment_reduce_map["arousal"] = "anxious"
    # sentiment_reduce_map["teasing"] = "anxious"
    # sentiment_reduce_map["mad"] = "grief"
    # sentiment_reduce_map["vulgar"] = "grief"
    # sentiment_reduce_map["negative"] = "grief"
    # sentiment_reduce_map["remorse"] = "grief"
    # sentiment_reduce_map["sad"] = "grief"
    # sentiment_reduce_map["not_funny"] = "worry"
    # sentiment_reduce_map["valence"] = "anticipation"
    # sentiment_reduce_map["trust"] = "anticipation"

    # # Merging the sentiment set 2
    # df['sentiment'] = df['sentiment'].map(sentiment_reduce_map)
    # df.dropna(inplace=True)
    # print("-" * 60)
    # print("After sentiment mapping Set_2")
    # print("Total number of sentiments: {}".format(df["sentiment"].nunique()))

    # after_sentiment_merge = df["sentiment"].value_counts().plot(kind="bar", figsize=(25, 10)).get_figure()
    # after_sentiment_merge.savefig(config.generic_path + 'after_sentiment_merge.jpeg')

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
    target_list = ['guilt', 'others', 'nocode', 'none', 'https',
                   'awkwardness', 'agreement', 'disagreement', 'emotion', 'empty', 'encouragement', 'motivate',
                   'optimism', 'racist/sexist']

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

    final_distribution = df_new_reduced["sentiment"].value_counts().plot(kind="bar", figsize=(25, 10)).get_figure()
    final_distribution.savefig(config.generic_path + 'final_distribution.jpeg')

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
    print("df_new_reduced[unique]: {}".format(df_new_reduced["sentiment"].unique))
    print("df_new_reduced[nunique]: {}".format(df_new_reduced["sentiment"].nunique()))

    return df_new_reduced, sentiment_map, sentiment_demap


def test_data_process(dataset):
    # reading the data frame
    # df = pd.read_csv("/content/drive/My Drive/AGJCSV/combained_validation_processed.csv")

    df = dataset  # encoding to avoid UnicodeDecodeError
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

    print("Total sentiment before mapping")
    print("Total number of sentiments: {}".format(df["sentiment"].nunique()))
    # Merging the sentiment set 1
    df['sentiment'] = df['sentiment'].map(sentiment_reduce_map)
    print("After sentiment mapping Set_1")
    print("Total number of sentiments: {}".format(df["sentiment"].nunique()))

    # New mapping
    sentiment_reduce_map["awe"] = "carving"
    sentiment_reduce_map["entertaining"] = "ridicule"
    sentiment_reduce_map["criticize"] = "ridicule"
    sentiment_reduce_map["sarcasm"] = "ridicule"
    sentiment_reduce_map["shame"] = "disgust"
    sentiment_reduce_map["enthusiasm"] = "surprise"
    sentiment_reduce_map["relief"] = "happy"
    sentiment_reduce_map["excitement"] = "happy"
    sentiment_reduce_map["love"] = "happy"
    sentiment_reduce_map["horror"] = "anxious"
    sentiment_reduce_map["pain"] = "anxious"
    sentiment_reduce_map["arousal"] = "anxious"
    sentiment_reduce_map["teasing"] = "anxious"
    sentiment_reduce_map["mad"] = "grief"
    sentiment_reduce_map["vulgar"] = "grief"
    sentiment_reduce_map["negative"] = "grief"
    sentiment_reduce_map["remorse"] = "grief"
    sentiment_reduce_map["sad"] = "grief"
    sentiment_reduce_map["not_funny"] = "worry"
    sentiment_reduce_map["valence"] = "anticipation"
    sentiment_reduce_map["trust"] = "anticipation"

    # Merging the sentiment set 2
    df['sentiment'] = df['sentiment'].map(sentiment_reduce_map)
    df.dropna(inplace=True)
    print("-" * 60)
    print("After sentiment mapping Set_2")
    print("Total number of sentiments: {}".format(df["sentiment"].nunique()))

    after_sentiment_merge = df["sentiment"].value_counts().plot(kind="bar", figsize=(25, 10)).get_figure()
    after_sentiment_merge.savefig(config.generic_path + 'after_sentiment_merge.jpeg')

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
    target_list = ['not-relevant', 'others', 'nocode', 'none', 'https',
                   'awkwardness', 'agreement', 'disagreement', 'emotion', 'empty', 'encouragement', 'motivate',
                   'optimism', 'racist/sexist']

    print("-" * 80)
    print("Sentiment to be dropped list, due to poor number of datapoints")
    print(target_list)

    def drop_target(target_list, df):
        """This is used to drop the particular sentiment"""
        print("df[df.sentiment.isin(target_list)].index: {}".format(df[df.sentiment.isin(target_list)].index))
        df_new = df.drop(df[df.sentiment.isin(target_list)].index)
        df_new.to_csv("df_new.csv")
        df_new_reduced = pd.read_csv("df_new.csv")
        return df_new_reduced

    # df_new_reduced is reduced target data frame
    df_new_reduced = drop_target(target_list, df)

    final_distribution = df_new_reduced["sentiment"].value_counts().plot(kind="bar", figsize=(25, 10)).get_figure()
    final_distribution.savefig(config.generic_path + 'final_distribution.jpeg')

    print("-" * 80)
    print("Total Number of reduced sentiment (number of targets)")
    print(df_new_reduced["sentiment"].nunique())
    df_new_reduced.dropna(inplace=True)

    # Encode mapping (categorical encoding of the target)
    sentiment_map = dict(zip(df_new_reduced['sentiment'], df_new_reduced['sentiment'].astype("category").cat.codes))
    sentiment_demap = dict(zip(df_new_reduced["sentiment"].astype("category").cat.codes, df_new_reduced["sentiment"]))
    df_new_reduced['sentiment'] = df_new_reduced['sentiment'].map(sentiment_map)

    print("-" * 80)
    print("df.shape: {}".format(df_new_reduced.shape))
    print("df.columns: {}".format(df_new_reduced.columns))

    return df_new_reduced[["sentence","sentiment"]], sentiment_map, sentiment_demap


def get_weight(df):
    """This will give weights to encounter imbalanced class problem"""

    # Getting number of data points for each class
    weight_count = df["sentiment"].value_counts(sort=False)
    # print(weight_count)

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


def save_model(EPOCH, model, optimizer, LOSS, ACCURACY, PATH):
    torch.save({
        'epoch': EPOCH,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': LOSS,
        'Accuracy': ACCURACY
    }, PATH)
    print("Saved the model")


def report(y_test, y_pred, sentiment_map):

    sorted_sentiment_map = list(sorted(sentiment_map.keys()))
    print("sorted_sentiment_map: {}".format(sorted_sentiment_map))
    print("len(sorted_sentiment_map): {}".format(len(sorted_sentiment_map)))
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=sorted_sentiment_map)
    confusion_matrix_df.columns = sorted_sentiment_map
    # display( confusion_matrix_df.style.background_gradient(cmap ='viridis') )

    report = classification_report(y_test, y_pred, output_dict=True, target_names=sorted_sentiment_map)
    print(report)
    print("*" * 120)
    print("Macro F1 score: {}".format(f1_score(y_test, y_pred, average="macro")))
    print("Micro F1 score: {}".format(f1_score(y_test, y_pred, average="micro")))
    print("Weighted F1 score: {}".format(f1_score(y_test, y_pred, average="weighted")))
    print("accuracy_score : {}".format(accuracy_score(y_test, y_pred)))
    print("*" * 120)

    # Printing the multiclass confusion matrix
    display(confusion_matrix_df.style.background_gradient(cmap='viridis', axis=1))

    return confusion_matrix_df.style.background_gradient(cmap='viridis', axis=1), report


def save_graph(graph_data, path=os.getcwd()):
    # graph_data is default dict
    with open(path + "graph_data.txt", "wb") as fp:
        print("graph_data.txt is saved to {}".format(path))
        pickle.dump(graph_data, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_graph(path=os.getcwd()):
    try:
        with open(path + "graph_data.txt", "rb") as fp:
            # pickle.dump(validate_data, fp)
            graph_dict = pickle.load(fp)

        return graph_dict
    except FileNotFoundError:
        print("No file named validate.txt")

    return "Nill"


def remove_unicode(x):
    # return "".join([a for a in x if a.isalnum()])
    x = x.encode('ascii', 'ignore').decode("utf-8")
    return x


def remove_link(x):
    x = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', x, flags=re.MULTILINE)
    return x


own_stopwords_small = ['a', 'on', 'that', 's', 't',
                       'own', 'same', 'so', 'than', 'too', 'very', 'an', 'the', 'and',
                       'because', 'as', 'until', 'while', 'of', 'at', 'by',
                       'who', 'whom', 'this', "that'll", 'these', 'those',
                       'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                       'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
                       'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
                       'for', 'with', 'about', 'against', 'between', 'into', 'through',
                       'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
                       'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
                       'then', 'once', 'here', 'there', 'when', 'where', ' why', 'how', 'all',
                       'any', 'both', 'other', 'some', 'such',
                       'no', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                       's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
                       'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'are', "aren't", 'couldn',
                       "couldn't", 'did', "didn't", 'does', "doesn't", 'had', "hadn't", ]


def remove_at_rate(x):
    string_list = []
    if len(x.split()) > 1:
        for string in x.split():
            if string[0] != "@":
                string_list.append(string)
            final = " ".join(string_list)
        # print(final)
        return final
    else:
        return x


def remove_stop(x, stop_words):
    string_list = []
    if len(x.split()) > 1:
        for string in x.split():
            if string not in stop_words:
                string_list.append(string)
            final = " ".join(string_list)
        # print(final)
        return final
    else:
        return x


puncts = [',', '.', '"', ':', ')', '(', '-', '|', ';', "'", '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',
          '~', '@', '£',
          '·', '_', '{', '}', '©', '^', '®', '`', '<', '→', '°', '€', '™', '›', '←', '×', '§', '″', '′', 'Â', '█', '½',
          'à', '…',
          '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥',
          '▓', '—', '‹', '─',
          '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾',
          'Ã', '⋅', '‘', '∞',
          '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', 'ï', 'Ø', '¹', '≤',
          '‡', '√']


def clean_text(x):
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, ' ')
    return " ".join(x.split())


contraction_dict = {"ain't": "is not", "aren't": "are not", "can't": "cannot",
                    "'cause": "because", "could've": "could have", "couldn't": "could not",
                    "didn't": "did not", "doesn't": "does not", "don't": "do not",
                    "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                    "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                    "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                    "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                    "i'll've": "i will have", "i'm": "i am", "i've": "i have", "isn't": "is not",
                    "it'd": "it would", "it'd've": "it would have", "it'll": "it will",
                    "it'll've": "it will have", "it's": "it is", "let's": "let us", "ma'am": "madam",
                    "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                    "mightn't've": "might not have",
                    "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                    "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock",
                    "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                    "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                    "she's": "she is", "should've": "should have", "shouldn't": "should not",
                    "shouldn't've": "should not have",
                    "so've": "so have", "so's": "so as", "this's": "this is", "that'd": "that would",
                    "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                    "there'd've": "there would have", "there's": "there is", "here's": "here is",
                    "they'd": "they would",
                    "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
                    "they're": "they are",
                    "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would",
                    "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                    "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                    "what're": "what are", "what's": "what is", "what've": "what have", "when's": "when is",
                    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have",
                    "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                    "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                    "won't've": "will not have",
                    "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                    "y'all": "you all",
                    "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                    "y'all've": "you all have",
                    "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                    "you'll've": "you will have", "you're": "you are",
                    "you've": "you have", "wasnt": "was not", "wont": "will not"}


def _get_contractions(contraction_dict):
    contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
    return contraction_dict, contraction_re


contractions, contractions_re = _get_contractions(contraction_dict)


def replace_contractions(text):
    def replace(match):
        return contractions[match.group(0)]

    return contractions_re.sub(replace, text)


def preprocess(x, stop_words=own_stopwords_small):
    x = str(x)  # converting input to string
    x = remove_link(x)  # Removing the link
    x = remove_at_rate(x)  # @name removal
    x = replace_contractions(x)
    # x = remove_unicode(x) # For emoji
    x = clean_text(x)  # Removing unwanted Punctuation, (?! r excluded)
    x = remove_stop(x, stop_words)  # Exculded some, like not
    return x

