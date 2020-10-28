# Setting up the device for GPU usage

import torch
from torch import cuda
import config
import utility
from tqdm import tqdm
import triage
import model
import pandas as pd
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader


print("-"*60)
device = 'cuda' if cuda.is_available() else 'cpu'
print("Available Device: {}".format(device))
print("-"*60)

model = model.DistillBERTClass()  # Creating the model shape
model.to(device)
checkpoint = torch.load(config.checkpoint_path)  # Loading the model from check point
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)  # Loading model to GPU

# Validation
# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()


def valid(model, testing_loader):
    model.eval()
    n_correct = 0;
    n_wrong = 0;
    total = 0
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    y_test_predicted = []
    # y_test_predicted_prob = []
    y_test_actual = []
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for _, data in enumerate(tqdm(testing_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            y_test_predicted_prob = softmax(outputs.data)
            print("y_test_predicted_prob: {}".format(y_test_predicted_prob))

            n_correct += utility.calculate_accuracy(big_idx, targets)
            print("y_test_predicted: {}".format(big_idx))
            print("type(y_test_predicted) : {}".format(type(big_idx)))
            print("y_test_actual: {}".format(targets))
            print("type(y_test_actual) : {}".format(type(targets)))
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

    return epoch_accu, y_test_actual, y_test_predicted


# Initiate the tokenizer
distill_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

test_params = {'batch_size': config.VALID_BATCH_SIZE,
               'shuffle': True,
               'num_workers': 0}


validation_frame_reduced, sentiment_map, sentiment_demap = utility.data_process(config.df_valid_path)
validation_frame_reduced.columns = ["TITLE", "ENCODE_CAT"]
validation_frame_reduced.reset_index(inplace=True)

validation_frame_reduced = validation_frame_reduced.sample(frac=0.01)

testing_set = triage.Triage(validation_frame_reduced, distill_tokenizer, config.MAX_LEN)
testing_loader = DataLoader(testing_set, **test_params)


print('This is the validation section to print the accuracy and see how it performs')
print('Here we are leveraging on the dataloader crearted for the validation dataset, the approach is using more of pytorch')

acc, y_test_actual, y_test_predicted=valid(model, testing_loader)

print("Accuracy on test data = %0.2f%%" % acc)

validation_confusion_matrix_df, classification_report = utility.report(y_test=y_test_actual,
                                                                       y_pred=y_test_predicted,
                                                                       sentiment_map=sentiment_map)

validation_confusion_matrix_df.to_excel(config.generic_path+"validation_confusion_matrix_df.xlsx")
classification_report.to_excel(config.generic_path+"classification_report.xlsx")