import torch
from transformers import BertModel
import config


# Creating the customized model,
# by adding a drop out and a dense layer on top of distil
# bert to get the final output for the model.


class BERTClass(torch.nn.Module):
    def __init__(self):
        super(DistillBERTClass, self).__init__()
        self.l1 = BertModel.from_pretrained(config.PRE_TRAINED_MODEL_NAME)
        self.pre_classifier = torch.nn.Linear(768, 768)  # O/P of the bert
        self.dropout = torch.nn.Dropout(0.3)  # Just a dropout
        self.classifier = torch.nn.Linear(768, 31)  # Since we combined sentiment to 31 targets
        # self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        # soft_output = self.softmax(output)
        return output
