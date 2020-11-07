# Defining some key variables that will be used later on in the training

MAX_LEN = 64  # 6081 datapoints are more than 60 length out of 2238471 which is just 0.2% of dataset
EPOCHS = 3
LEARNING_RATE = 1e-05  # 5e-5, 3e-5, 2e-5
PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased'

df_path = r"/content/drive/My Drive/AGJCSV/comabined_bert_df.csv"  # (2244552, 2)
# generic_path = r"/content/drive/My Drive/AGJCSV/models/bert_31_1epoch_unbalanced/"
# df_valid_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed_bert.csv"


TRAIN_BATCH_SIZE = 16  # 8,16,32,64
VALID_BATCH_SIZE = 16  # 8,16,32,64
TEST_BATCH_SIZE = 16  # 8,16,32,64
checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced/pytorch_distilbert_31_unbalanced.pt"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced/"


# v1 model uses, without weight, train batch size of 4 | Fixing the Learning rate constant
TRAIN_BATCH_SIZE = 4  # 8,16,32,64
VALID_BATCH_SIZE = 4  # 8,16,32,64
TEST_BATCH_SIZE = 4  # 8,16,32,64
checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced_v1/pytorch_distilbert_31_unbalanced_v1.pt"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced_v1/"


# v2 model uses, without weight, train batch size of 64 | fixing the Learning rate constant
TRAIN_BATCH_SIZE = 64  # 8,16,32,64
VALID_BATCH_SIZE = 64  # 8,16,32,64
TEST_BATCH_SIZE = 64  # 8,16,32,64
checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced_v2/pytorch_distilbert_31_unbalanced_v2.pt"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced_v2/"

