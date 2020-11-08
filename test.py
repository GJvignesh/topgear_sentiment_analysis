# Setting up the device for GPU usage
import torch
from torch import cuda, nn
from tqdm import tqdm
import pandas as pd
import pickle
from transformers import DistilBertTokenizer, AdamW
from torch.utils.data import Dataset, DataLoader

import config
import utility
import triage
import model
import prepare_data

print("-" * 60)
device = 'cuda' if cuda.is_available() else 'cpu'
print("Available Device: {}".format(device))
print("-" * 60)


def valid(model, testing_loader, loss_fn):
    model.eval()
    n_correct = 0;
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    y_test_predicted = []
    y_test_predicted_prob_list = []
    y_test_actual = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            # print("OUTPUTS: {}".format(outputs))
            # print("targets: {}".format(targets))
            loss = loss_fn(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            y_test_predicted_prob = softmax(outputs.data)
            # print("y_test_predicted_prob: {}".format(y_test_predicted_prob))
            y_test_predicted_prob_list.append(y_test_predicted_prob.tolist())

            n_correct += utility.calculate_accuracy(big_idx, targets)
            # print("y_test_predicted: {}".format(big_idx))
            # print("type(y_test_predicted) : {}".format(type(big_idx)))
            # print("y_test_actual: {}".format(targets))
            # print("type(y_test_actual) : {}".format(type(targets)))
            y_test_predicted.extend(big_idx.tolist())
            y_test_actual.extend(targets.tolist())
            # print(y_test_predicted)
            # print(y_test_actual)
            # print(len(y_test_predicted))
            # print(len(y_test_actual))
            # print("*"*120)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _ % 5000 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f"Validation Loss per 100 steps: {loss_step}")
                print(f"Validation Accuracy per 100 steps: {accu_step}")
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Validation Loss Epoch: {epoch_loss}")
    print(f"Validation Accuracy Epoch: {epoch_accu}")

    return epoch_loss, epoch_accu, y_test_actual, y_test_predicted, y_test_predicted_prob_list


##################################################################

# Prepare the data
df_path = config.df_path

df = pd.read_csv(df_path, encoding="ISO-8859-1")
print("df.shape: {}".format(df.shape))

# Preprocess the data
print("Cleaning the data: removing, punctuation, stopwords, https, contradiction")
test_data = df["sentence"].apply(utility.preprocess)
test_data["sentiment"] = df["sentiment"]
test_data.dropna(inplace=True)

print("type(test_data): {}".format(type(test_data)))
print("test_data.shape: {}".format(test_data.shape))

# This will give reduced sentiment [FYI: Its excepting preprocessed dataframe]
df_new_reduced, sentiment_map, sentiment_demap = utility.test_data_process(dataset=test_data)


# Initiate the tokenizer
bert_tokenizer = DistilBertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

# Creating instance of Preprocess
# This Preprocess internally Triage class
# This will split data and encode using passing tokenizer
# Creating instance of the class
Preprocess = prepare_data.Preprocess(dataframe=df_new_reduced,
                                     tokenizer=bert_tokenizer,
                                     max_len=config.MAX_LEN,
                                     train_batch_size=config.TRAIN_BATCH_SIZE,
                                     valid_batch_size=config.VALID_BATCH_SIZE,
                                     test_batch_size=config.TEST_BATCH_SIZE)

# Accessing the process_data_for_model method of Preprocess class
testing_loader = Preprocess.process_data_for_test()

#################################################################################
model = model.DistillBERTClass()  # Creating the model shape
model.to(device)

# Loading back the model from checkpoint
checkpoint = torch.load(config.checkpoint_path, map_location=device)  # Loading the model from check point
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)  # Loading model to GPU

# Validation on test data
# Creating the loss function
# Optimizer is not needed since its for prediction
loss_function = torch.nn.CrossEntropyLoss()

test_loss, test_accu, y_test_actual, y_test_predicted, y_test_predicted_prob_list = valid(model=model,
                                                                                          testing_loader=testing_loader,
                                                                                          loss_fn=loss_function)

print("Loss on test data = %0.2f%%" % test_loss)
print("Accuracy on test data = %0.2f%%" % test_accu)


test_confusion_matrix_df, classification_report = utility.report(y_test=y_test_actual,
                                                                       y_pred=y_test_predicted,
                                                                       sentiment_map=sentiment_map)


test_confusion_matrix_df.to_excel(config.generic_path + "test_confusion_matrix_df.xlsx")
classification_report_df = pd.DataFrame(classification_report).transpose()
classification_report_df.to_csv(config.generic_path + "test_classification_report.csv")
