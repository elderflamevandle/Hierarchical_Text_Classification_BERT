import torch
from torch import nn
from transformers import BertModel

class HierarchicalProductClassifier(nn.Module):
    def __init__(self, n_classes1, n_classes2, n_classes3):
        super(HierarchicalProductClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, n_classes1)
        self.fc2 = nn.Linear(self.bert.config.hidden_size + n_classes1, n_classes2)
        self.fc3 = nn.Linear(self.bert.config.hidden_size + n_classes1 + n_classes2, n_classes3)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        output = self.drop(pooled_output)

        cat1_output = self.fc1(output)
        cat1_probs = torch.softmax(cat1_output, dim=1)

        cat2_input = torch.cat((output, cat1_probs), dim=1)
        cat2_output = self.fc2(cat2_input)
        cat2_probs = torch.softmax(cat2_output, dim=1)

        cat3_input = torch.cat((output, cat1_probs, cat2_probs), dim=1)
        cat3_output = self.fc3(cat3_input)

        return cat1_output, cat2_output, cat3_output