# Importing the libraries needed
from time import gmtime, strftime
from tqdm import tqdm
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch import cuda, nn
from transformers import DistilBertTokenizer, AdamW
from collections import defaultdict

import model
import prepare_data
import config
import utility
import telegram_bot as bot


# Setting up the device for GPU usage
print("-" * 60)
device = 'cuda' if cuda.is_available() else 'cpu'
print("Available Device: {}".format(device))
print("-" * 60)


# Training
# Defining the training function on the 80% of the dataset for tuning the distilbert model
def train(model, training_loader, loss_fn, optimizer):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    bot.telegram_bot_sendtext("training started for : " + config.generic_path)
    for _, data in enumerate(tqdm(training_loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)

        # print("len(targets): ".format(len(targets)))
        # print("len(ids): ".format(len(targets)))
        # print("len(ids[0]): ".format(len(ids[0])))

        # print("*"*120)
        # print("ids: {}".format(ids))
        # print("mask: {}".format(mask))
        # print("targets: {}".format(targets))
        # print("*"*120)

        # Calling the created model
        # outputs, probability = model(ids, mask)
        outputs = model(ids, mask)
        # print("MODEL OUTPUTS: {}".format(outputs))
        # print("MODEL probability: {}".format(probability))
        loss = loss_fn(outputs, targets)
        # print("loss: {}".format(loss))
        tr_loss += loss.item()
        # print("loss.item(): {}".format(loss.item()))
        # print("outputs.data: {}".format(outputs.data))
        # print("torch.max(outputs.data, dim=1): {}".format(torch.max(outputs.data, dim=1)))
        big_val, big_idx = torch.max(outputs.data, dim=1)
        # print("big_idx: {}".format(big_idx))
        n_correct += utility.calculate_accuracy(big_idx, targets)
        # print("+"*120)

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        if _ % 5000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            # bot.telegram_bot_sendtext(config.generic_path)
            # bot.telegram_bot_sendtext("Training Loss per 5000 steps: " + str(loss_step))
            # bot.telegram_bot_sendtext("Training Accuracy per 5000 steps: " + str(accu_step))
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # To avoid gradient explode
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    bot.telegram_bot_sendtext(config.generic_path)
    bot.telegram_bot_sendtext("Final Training Loss at end of one Epoch " + str(epoch_loss))
    bot.telegram_bot_sendtext("Final Training Accuracy at end of one Epoch: " + str(epoch_accu))
    bot.telegram_bot_sendtext("1 EPOCH completed !!")

    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return epoch_loss, epoch_accu


def eval_model(model, valid_loader, loss_fn):
    model.eval()
    n_correct = 0;
    tr_loss = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    y_test_predicted = []
    # y_test_predicted_prob_list = []
    y_test_actual = []
    # softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        for _, data in enumerate(tqdm(valid_loader, 0)):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            # print("OUTPUTS: {}".format(outputs))
            # print("targets: {}".format(targets))
            loss = loss_fn(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            # y_test_predicted_prob = softmax(outputs.data)
            # print("y_test_predicted_prob: {}".format(y_test_predicted_prob))
            # y_test_predicted_prob_list.append(y_test_predicted_prob.tolist())

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

    return epoch_loss, epoch_accu, y_test_actual, y_test_predicted


##################################################################

# Prepare the data
df_path = config.df_path

# This will give reduced sentiment [FYI: Its excepting preprocessed dataframe]
df_new_reduced, sentiment_map, sentiment_demap = utility.data_process(dataset_path=df_path)


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
training_loader, valid_loader, testing_loader = Preprocess.process_data_for_model()

model = model.DistillBERTClass()  # instance of the bert own model
model.to(device)  # passing to GPU

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()  # without weight vector (to increase the accuracy score)

optimizer = AdamW(params=model.parameters(), lr=config.LEARNING_RATE,
                  correct_bias=False)  # recommended optimizer for Bert

# optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)


graph = defaultdict(list)
best_validation_accuracy = 0

for epoch in range(config.EPOCHS):
    train_epoch_loss, train_epoch_accu = train(model=model, training_loader=training_loader,
                                               loss_fn=loss_function, optimizer=optimizer)

    valid_epoch_loss, valid_epoch_accu, y_valid_actual, y_valid_predicted = eval_model(model=model,
                                                                                       valid_loader=valid_loader,
                                                                                       loss_fn=loss_function)

    validation_confusion_matrix_df, classification_report = utility.report(y_test=y_valid_actual,
                                                                     y_pred=y_valid_predicted,
                                                                     sentiment_map=sentiment_map)

    validation_confusion_matrix_df.to_excel(config.generic_path + "validation_confusion_matrix_df.xlsx")
    classification_report_df = pd.DataFrame(classification_report).transpose()
    classification_report_df.to_csv(config.generic_path + "validation_classification_report.csv")

    graph["train_epoch_loss_list"].append(train_epoch_loss)
    graph['train_epoch_accu_list'].append(train_epoch_accu)
    graph['valid_epoch_loss_list'].append(valid_epoch_loss)
    graph['valid_epoch_accu_list'].append(valid_epoch_accu)

    validation_f1_score_macro = f1_score(y_valid_actual, y_valid_predicted, average="macro")

    graph['validation_f1_score_macro_list'].append(validation_f1_score_macro)

    print("validation_f1_score_macro: {}".format(validation_f1_score_macro))
    if valid_epoch_accu > best_validation_accuracy:

        # Creating check point
        utility.save_model(EPOCH=epoch, model=model, optimizer=optimizer,
                           LOSS=train_epoch_loss, ACCURACY=train_epoch_accu,
                           PATH=config.checkpoint_path)

        best_validation_accuracy = valid_epoch_accu
        graph["best_validation_accuracy"] = best_validation_accuracy

    print("graph: {}".format(graph))
    utility.save_graph(graph_data= graph, path=config.generic_path)
    bot.telegram_bot_sendtext(graph)
