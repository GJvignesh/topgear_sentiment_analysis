# Defining some key variables that will be used later on in the training

MAX_LEN = 50  # 486 datapoints are more than 50 length out of 1488909 which is just 0.03264135014295702 %
TRAIN_BATCH_SIZE = 16  # 8,16,32,64
VALID_BATCH_SIZE = 16  # 8,16,32,64
TEST_BATCH_SIZE = 16  # 8,16,32,64
EPOCHS = 3
LEARNING_RATE = 1e-05  # 5e-5, 3e-5, 2e-5
PRE_TRAINED_MODEL_NAME = 'distilbert-base-cased'


# checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced/pytorch_distilbert_31_unbalanced.pt"

# Un-Balanced
# checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalancedn/pytorch_distilbert_31_unbalanced_fulln.pt"
# generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalancedn/"
# df_path = r"/content/drive/My Drive/AGJCSV/processed_new.csv"  # (1488909, 2)

# Balanced
checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_balancedn/pytorch_distilbert_31_balanced_fulln.pt"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_balancedn/"
df_path = r"/content/drive/My Drive/AGJCSV/processed_new.csv"  # (1488909, 2)

df_valid_path = r"/content/drive/My Drive/AGJCSV/Validation_NEW/Validation_NEW.csv"

