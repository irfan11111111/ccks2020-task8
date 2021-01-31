# -*- coding: utf-8 -*-
# @Time    : 2020/2/22 4:47 下午
# @Author  : lizhen
# @FileName: model.py
# @Description:
import torch.nn as nn
import torch.nn.functional as F
import torch
from base import BaseModel
from operator import itemgetter
from transformers import BertModel, XLNetModel
from utils.model_utils import prepare_pack_padded_sequence, matrix_mul, element_wise_mul
from .torch_crf_r import CRF
import numpy as np
from .lstm import LSTM
from model.modeling_mobilebert import MobileBertConfig, MobileBertForPreTraining, load_tf_weights_in_mobilebert,MobileBertModel
import time



class Bert_CRF(BaseModel):

    def __init__(self, bert_path, bert_train, num_tags, dropout, restrain,model_name="mobilebert"):
        super(Bert_CRF, self).__init__()
        if model_name=="mobilebert":
            # 改为mobilebert——config路径
            # config = MobileBertConfig.from_json_file("./pretrained/chinese_roberta_wwm_ext_pytorch/config.json")
            # print("Building PyTorch model from configuration: {}".format(str(config)))
            self.bert = MobileBertModel.from_pretrained(bert_path)
        else:
            self.bert = BertModel.from_pretrained(bert_path)
        # print(self.bert)
        self.lstm = LSTM(512, num_layers=1, hidden_size=768,
                             bidirectional=True,
                             batch_first=True)

        self.dropout = nn.Dropout(dropout)
        # self.fc = nn.Linear(768 * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True, restrain_matrix=restrain, loss_side=2.5)
        # 对bert进行训练
        for name, param in self.bert.named_parameters():
            param.requires_grad = bert_train

        self.fc_tags = nn.Linear(1536, num_tags)

    def forward(self, context, mask_bert,seq_len):
        # context  输入的句子序列
        # seq_len  句子长度
        # mask     对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]

        # cls [batch_size, 768]
        # sentence [batch size,sen len,  768]
        bert_sentence, bert_cls = self.bert(context, attention_mask=mask_bert)
        sentence_len = bert_sentence.shape[1]

        # bert_cls = bert_cls.unsqueeze(dim=1).repeat(1, sentence_len, 1)
        # bert_sentence = bert_sentence + bert_cls
        bert_sentence=self.dropout(bert_sentence)
        feats, _ = self.lstm(bert_sentence, seq_len=seq_len)
        # feats = self.fc(feats)
        # feats = self.dropout(feats)
        pred_tags = self.fc_tags(feats)[:, 1:, :]
        return pred_tags