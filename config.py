# Defining some key variables that will be used later on in the training
MAX_LEN = 300
TRAIN_BATCH_SIZE = 70
VALID_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-05

checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_balanced/pytorch_distilbert_31_balanced.pt"
# df_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed.csv"
df_path = r"/content/drive/My Drive/AGJCSV/combained_train_processed_new.csv"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_balanced/"
df_valid_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed_new.csv"
