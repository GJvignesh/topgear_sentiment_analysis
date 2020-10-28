# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 25
VALID_BATCH_SIZE = 250
EPOCHS = 1
LEARNING_RATE = 1e-05

checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_32_balanced/pytorch_distilbert_32_balanced.pt"
df_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed.csv"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_32_balanced/"