import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import Triage
import utility


# Mounting the google drive
from google.colab import drive
drive.mount("/content/drive")


class Preprocess(Triage.Triage):

    def __init__(self, dataframe, tokenizer, max_len, train_batch_size, valid_batch_size):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size

    def process_data_for_model(self):
        # Split the data
        # Train-test split
        print("Splitting the data")

        # Taking sentence and sentiment alone
        train_dataset, test_dataset = train_test_split(self.data[["sentence", "sentiment"]], train_size=0.7,
                                                       random_state=1)

        print("train_dataset.shape: {}".format(train_dataset.shape))
        print("test_dataset.shape: {}".format(test_dataset.shape))

        # column renaming
        train_dataset.columns = ["TITLE", "ENCODE_CAT"]
        test_dataset.columns = ["TITLE", "ENCODE_CAT"]

        # resetting the index
        train_dataset.reset_index(inplace=True)
        test_dataset.reset_index(inplace=True)

        # Here we are calling Triage class which inherits the Dataset class
        training_set = Triage.Triage(train_dataset, self.tokenizer, self.max_len)
        testing_set = Triage.Triage(test_dataset, self.tokenizer, self.max_len)

        train_params = {'batch_size': self.train_batch_size,
                        'shuffle': True,
                        'num_workers': 0
                        }

        test_params = {'batch_size': self.valid_batch_size,
                       'shuffle': True,
                       'num_workers': 0
                       }

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        return training_loader, testing_loader
