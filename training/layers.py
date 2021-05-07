"""Assortment of layers for use in models.py."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistilBERTClassifier(nn.Module):
    """Classification layer on top of the DistilBERT model
    
    """
    def __init__(self, hidden_size, num_labels, drop_prob, use_img, img_size):
        super(DistilBERTClassifier, self).__init__()
        self.drop_prob = drop_prob
        self.linear_1 = nn.Linear(hidden_size, num_labels)
        # self.use_img = use_img
        # if use_img:
        #     self.linear_2 = nn.Linear(img_size, num_labels)
        
    def forward(self, con_x):
        con_x = F.dropout(con_x, self.drop_prob, self.training)
        x = self.linear_1(con_x)
        # if self.use_img:
        #     img_x = F.dropout(img_x, self.drop_prob, self.training)
        #     x += self.linear_2(img_x)
        
        return x
        
    
