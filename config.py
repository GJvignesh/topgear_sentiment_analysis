# Defining some key variables that will be used later on in the training
MAX_LEN = 300
TRAIN_BATCH_SIZE = 70
VALID_BATCH_SIZE = 1
EPOCHS = 1
LEARNING_RATE = 1e-05

checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced_fulldata/pytorch_distilbert_31_unbalanced_full.pt"
# df_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed.csv"
df_path = r"/content/drive/My Drive/AGJCSV/final_train_df.csv"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_31_unbalanced_fulldata/"
df_valid_path = r"/content/drive/My Drive/AGJCSV/final_test_df.csv"
