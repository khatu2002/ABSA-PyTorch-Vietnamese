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
        # Kiểm tra kiểu dữ liệu đầu vào
        print("Kiểu của text_bert_indices:", type(text_bert_indices))
        print("Kiểu của bert_segments_ids:", type(bert_segments_ids))
        
        # Lấy đầu ra từ mô hình BERT
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        
        # Kiểm tra kiểu dữ liệu đầu ra từ BERT
        print("Kiểu của pooled_output trước dropout:", type(pooled_output))
        
        pooled_output = self.dropout(pooled_output)
        
        # Kiểm tra kiểu dữ liệu đầu ra sau dropout
        print("Kiểu của pooled_output sau dropout:", type(pooled_output))
        
        logits = self.dense(pooled_output)
        return logits
