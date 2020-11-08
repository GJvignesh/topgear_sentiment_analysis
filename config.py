# Defining some key variables that will be used later on in the training

MAX_LEN = 64  # 6081 datapoints are more than 60 length out of 2238471 which is just 0.2% of dataset
TRAIN_BATCH_SIZE = 16  # 8,16,32,64
VALID_BATCH_SIZE = 16  # 8,16,32,64
TEST_BATCH_SIZE = 16  # 8,16,32,64
EPOCHS = 3
LEARNING_RATE = 1e-05  # 5e-5, 3e-5, 2e-5
PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased'


# checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced/pytorch_distilbert_31_unbalanced.pt"


checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced/pytorch_distilbert_31_unbalanced_full.pt"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced/"
df_path = r"/content/drive/My Drive/AGJCSV/comabined_bert_df.csv"  # (2244552, 2)


# generic_path = r"/content/drive/My Drive/AGJCSV/models/bert_31_1epoch_unbalanced/"
# df_valid_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed_bert.csv"


