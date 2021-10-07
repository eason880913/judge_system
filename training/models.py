"""Model classes"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import layers

from transformers import DistilBertModel, AlbertModel, DistilBertConfig

class DistilBERT(nn.Module):
    """DistilBERT model to classify news

    Based on the paper:
    DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
    by Victor Sanh, Lysandre Debut, Julien Chaumond, Thomas Wolf
    (https://arxiv.org/abs/1910.01108)
    """
    def __init__(self, hidden_size, num_labels, drop_prob, freeze):
        super(DistilBERT, self).__init__()
        config = DistilBertConfig(vocab_size=119547)
        self.distilbert = DistilBertModel(config)
        for param in self.distilbert.parameters():
            param.requires_grad = not freeze
        self.classifier = layers.DistilBERTClassifier(hidden_size, num_labels,
                                                      drop_prob=drop_prob)

    def forward(self, input_idxs, atten_masks):
        con_x = self.distilbert(input_ids=input_idxs,
                                attention_mask=atten_masks)[0][:, 0]
        logit = self.classifier(con_x)
        log = torch.sigmoid(logit)

        return log

