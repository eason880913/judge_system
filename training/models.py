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
    def __init__(self, hidden_size, num_labels, drop_prob, freeze, use_img, img_size):
        super(DistilBERT, self).__init__()
        self.img_size = img_size
        self.use_img = use_img
        config = DistilBertConfig(vocab_size=119547)
        self.distilbert = DistilBertModel(config)
        for param in self.distilbert.parameters():
            param.requires_grad = not freeze
        self.classifier = layers.DistilBERTClassifier(hidden_size, num_labels,
                                                      drop_prob=drop_prob,
                                                      use_img=use_img,
                                                      img_size=img_size)

    def forward(self, input_idxs, atten_masks):
        con_x = self.distilbert(input_ids=input_idxs,
                                attention_mask=atten_masks)[0][:, 0]
        # img_x = self.resnet18(images).view(-1, self.img_size) if self.use_img else None
        logit = self.classifier(con_x)
        log = torch.sigmoid(logit)

        return log


class ALBERT(nn.Module):
    """ALBERT model to classify news

    Based on the paper:
    ALBERT: A Lite BERT for Self-supervised Learning of Language Representations
    by Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, Radu Soricut
    (https://arxiv.org/abs/1909.11942)
    """
    def __init__(self, hidden_size, num_labels, drop_prob, freeze, use_img, img_size):
        super(ALBERT, self).__init__()
        self.img_size = img_size
        resnet18 = torchvision.models.resnet18(pretrained=True)
        self.resnet18 = nn.Sequential(*list(resnet18.children())[:-1])
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.albert = AlbertModel.from_pretrained('voidful/albert_chinese_base')
        for param in self.albert.parameters():
            param.requires_grad = not freeze
        self.classifier = layers.DistilBERTClassifier(hidden_size, num_labels,
                                                      drop_prob=drop_prob,
                                                      use_img=use_img,
                                                      img_size=img_size)

    def forward(self, input_idxs, atten_masks, images):
        con_x = self.albert(input_ids=input_idxs,
                            attention_mask=atten_masks)[0][:, 0]
        img_x = self.resnet18(images).view(-1, self.img_size)
        logit = self.classifier(con_x, img_x)
        log = F.log_softmax(logit, dim=1)

        return log