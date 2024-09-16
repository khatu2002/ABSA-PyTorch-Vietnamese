import torch
import torch.nn as nn

class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]

        # Lấy đầu ra từ mô hình BERT
        outputs = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)

        # Thay đổi này để phù hợp với cấu trúc của đầu ra
        pooled_output = outputs.pooler_output  # Dựa vào cấu trúc của outputs

        pooled_output = self.dropout(pooled_output)
        
        logits = self.dense(pooled_output)
        return logits
