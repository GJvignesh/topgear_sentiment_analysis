# Importing the libraries needed
from time import gmtime, strftime
from tqdm import tqdm
import pandas as pd
import torch
# import transformers
# from torch.utils.data import Dataset, DataLoader
# from transformers import DistilBertModel, DistilBertTokenizer
from torch import cuda, nn
from transformers import BertTokenizer, AdamW
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

# Prepare the data
df_path = config.df_path
df_new_reduced, sentiment_map, sentiment_demap = utility.data_process(dataset_path=df_path)

# df_new_reduced sentiment is already encoded
class_weight = utility.get_weight(df_new_reduced)

# Initiate the tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(config.PRE_TRAINED_MODEL_NAME)

# Creating instance of Preprocess
# This Preprocess internally Triage class
# This will split data and encode using passing tokenizer
# Creating instance of the class
Preprocess = prepare_data.Preprocess(dataframe=df_new_reduced,
                                     tokenizer=bert_tokenizer,
                                     max_len=config.MAX_LEN,
                                     train_batch_size=config.TRAIN_BATCH_SIZE,
                                     valid_batch_size=config.VALID_BATCH_SIZE)

# Accessing the process_data_for_model method of Preprocess class
training_loader, testing_loader = Preprocess.process_data_for_model()

model = model.BERTClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss(weight=class_weight.to(device))
print("Class Weight: {}".format(class_weight))
optimizer = AdamW(params = model.parameters(), lr=config.LEARNING_RATE, correct_bias=False)
# optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)


# Training

# Defining the training function on the 80% of the dataset for tuning the distilbert model
def train(epoch):
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
        loss = loss_function(outputs, targets)
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
            bot.telegram_bot_sendtext(config.generic_path)
            bot.telegram_bot_sendtext("Training Loss per 5000 steps: " + str(loss_step))
            bot.telegram_bot_sendtext("Training Accuracy per 5000 steps: " + str(accu_step))
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")
            print(strftime("%Y-%m-%d %H:%M:%S", gmtime()))

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples

    # Creating check point
    utility.save_model(EPOCH=epoch, model=model, optimizer=optimizer,
                       LOSS=epoch_loss, ACCURACY=epoch_accu,
                       PATH=config.checkpoint_path)

    bot.telegram_bot_sendtext(generic_path)
    bot.telegram_bot_sendtext("Final Training Loss Epoch " + str(epoch_loss))
    bot.telegram_bot_sendtext("Final Training Accuracy Epoch: " + str(epoch_accu))
    bot.telegram_bot_sendtext("EPOCH completed")
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return


for epoch in range(config.EPOCHS):
    train(epoch)