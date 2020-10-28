# Defining some key variables that will be used later on in the training
MAX_LEN = 300
TRAIN_BATCH_SIZE = 80
VALID_BATCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05

checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_32_unbalanced/pytorch_distilbert_31_unbalanced.pt"
df_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed.csv"
generic_path = r"/content/drive/My Drive/AGJCSV/models/distill_bert_32_unbalanced/"