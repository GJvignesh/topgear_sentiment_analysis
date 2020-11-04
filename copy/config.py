# Defining some key variables that will be used later on in the training
MAX_LEN = 64
TRAIN_BATCH_SIZE = 16  # 16,32,64
VALID_BATCH_SIZE = 16  # 16,32,64
EPOCHS = 1
LEARNING_RATE = 2e-05  # 5e-5, 3e-5, 2e-5
PRE_TRAINED_MODEL_NAME = 'bert-base-cased'

checkpoint_path = r"/content/drive/My Drive/AGJCSV/models/bert_31_1epoch/pytorch_bert_31_balanced.pt"
df_path = r"C:/Users/vgopalja.PARTNERS/bert_multiclass/combained_train_processed_bert.csv"
# df_path = r"/content/drive/My Drive/AGJCSV/combained_train_processed_new.csv"
generic_path = r"/content/drive/My Drive/AGJCSV/models/bert_31_1epoch/"
df_valid_path = r"/content/drive/My Drive/AGJCSV/combained_validation_processed_bert.csv"
