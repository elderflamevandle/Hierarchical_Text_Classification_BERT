import torch
from torch import nn
from transformers import AutoModel

class HierarchicalProductClassifier(nn.Module):
    def __init__(self, n_classes_list, model_name='bert-base-uncased', dropout_p=0.3):
        """
        n_classes_list: List of integers representing the number of classes at each hierarchy level.
        model_name: HuggingFace model backbone name.
        """
        super(HierarchicalProductClassifier, self).__init__()

        self.backbone = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=dropout_p)
        self.n_levels = len(n_classes_list)
        
        # Dynamically create classification heads
        self.classifiers = nn.ModuleList()
        current_input_size = self.backbone.config.hidden_size
        
        for i, num_classes in enumerate(n_classes_list):
            self.classifiers.append(nn.Linear(current_input_size, num_classes))
            # The input to the next level is the hidden size + the probabilities of all previous levels
            current_input_size += num_classes

    def forward(self, input_ids, attention_mask):
        # Handle models that might return different tuple structures
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Use pooler_output if available, else take representation of [CLS] token
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        output = self.drop(pooled_output)

        level_outputs = []
        current_input = output
        
        for i in range(self.n_levels):
            level_output = self.classifiers[i](current_input)
            level_outputs.append(level_output)
            
            # For the next level, append the softmax probabilities of the current level
            if i < self.n_levels - 1:
                level_probs = torch.softmax(level_output, dim=1)
                current_input = torch.cat((current_input, level_probs), dim=1)

        return level_outputs  # Returns a list of outputs for each level