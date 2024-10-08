import torch
import torch.nn.functional as F
import argparse
import numpy as np
#import spacy
import re

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, pad_and_truncate
from models import LSTM, IAN, MemNet, RAM, TD_LSTM, TC_LSTM, Cabasc, ATAE_LSTM, TNet_LF, AOA, MGAN, ASGCN, LCF_BERT
from models.aen import CrossEntropyLoss_LSR, AEN_BERT
from models.bert_spc import BERT_SPC
from dependency_graph import dependency_adj_matrix

from transformers import BertModel
import argparse

class Inferer:
    """A simple inference example with aspect extraction"""
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
        else:
            self.tokenizer = build_tokenizer(
                fnames=[opt.dataset_file['train'], opt.dataset_file['test']],
                max_seq_len=opt.max_seq_len,
                dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
            embedding_matrix = build_embedding_matrix(
                word2idx=self.tokenizer.word2idx,
                embed_dim=opt.embed_dim,
                dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(opt.embed_dim), opt.dataset))
            self.model = opt.model_class(embedding_matrix, opt)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        self.model = self.model.to(opt.device)
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)
        # Load spaCy model for aspect extraction
        #self.nlp = spacy.load('en_core_web_sm')  # Replace with appropriate Vietnamese model if needed


    def extract_aspect(self, sentence):

        # Define known aspect-related words or patterns
        aspect_keywords = ['giao hàng', 'đóng gói', 'chất lượng', 'giá', 'dung lượng', 'màn hình', 'pin', 'hàng', 'sản phẩm', 'máy', 'phục vụ', 'hộp']

        # Sort aspect keywords by length in descending order to prioritize longer phrases first
        aspect_keywords = sorted(aspect_keywords, key=len, reverse=True)

        # Tokenize the sentence
        sentence_lower = sentence.lower()

        # Create a list to store found aspects
        found_aspects = []

        # Search for aspect keywords in the sentence
        for keyword in aspect_keywords:
            if re.search(rf'\b{keyword}\b', sentence_lower):
                # If the keyword is not a part of any already found aspect, add it
                if not any(keyword in aspect for aspect in found_aspects):
                    found_aspects.append(keyword)

        # Return the list of found aspects
        return found_aspects if found_aspects else None

    def evaluate(self, text, aspect):
        aspect = aspect.lower().strip()
        text_left, _, text_right = [s.strip() for s in text.lower().partition(aspect)]
        
        text_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        context_indices = self.tokenizer.text_to_sequence(text_left + " " + text_right)
        left_indices = self.tokenizer.text_to_sequence(text_left)
        left_with_aspect_indices = self.tokenizer.text_to_sequence(text_left + " " + aspect)
        right_indices = self.tokenizer.text_to_sequence(text_right, reverse=True)
        right_with_aspect_indices = self.tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
        aspect_indices = self.tokenizer.text_to_sequence(aspect)
        left_len = np.sum(left_indices != 0)
        aspect_len = np.sum(aspect_indices != 0)
        aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

        text_len = np.sum(text_indices != 0)
        concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
        concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
        concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

        text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
        aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

        dependency_graph = dependency_adj_matrix(text)

        data = {
            'concat_bert_indices': concat_bert_indices,
            'concat_segments_indices': concat_segments_indices,
            'text_bert_indices': text_bert_indices,
            'aspect_bert_indices': aspect_bert_indices,
            'text_indices': text_indices,
            'context_indices': context_indices,
            'left_indices': left_indices,
            'left_with_aspect_indices': left_with_aspect_indices,
            'right_indices': right_indices,
            'right_with_aspect_indices': right_with_aspect_indices,
            'aspect_indices': aspect_indices,
            'aspect_boundary': aspect_boundary,
            'dependency_graph': dependency_graph,
        }

        t_inputs = [torch.tensor([data[col]], device=self.opt.device) for col in self.opt.inputs_cols]
        t_outputs = self.model(t_inputs)
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()

        return t_probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentence', type=str, required=True, help='Input sentence for aspect extraction and sentiment analysis')
    args = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'td_lstm': TD_LSTM,
        'tc_lstm': TC_LSTM,
        'atae_lstm': ATAE_LSTM,
        'ian': IAN,
        'memnet': MemNet,
        'ram': RAM,
        'cabasc': Cabasc,
        'tnet_lf': TNet_LF,
        'aoa': AOA,
        'mgan': MGAN,
        'asgcn': ASGCN,
        'bert_spc': BERT_SPC,
        'aen_bert': AEN_BERT,
        'lcf_bert': LCF_BERT,
    }
    dataset_files = {
        'comment': {
            'train': './datasets/comment/train.raw',
            'test': './datasets/comment/test.raw'
        }
    }
    input_colses = {
        'lstm': ['text_indices'],
        'td_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices'],
        'tc_lstm': ['left_with_aspect_indices', 'right_with_aspect_indices', 'aspect_indices'],
        'atae_lstm': ['text_indices', 'aspect_indices'],
        'ian': ['text_indices', 'aspect_indices'],
        'memnet': ['context_indices', 'aspect_indices'],
        'ram': ['text_indices', 'aspect_indices', 'left_indices'],
        'cabasc': ['text_indices', 'aspect_indices', 'left_with_aspect_indices', 'right_with_aspect_indices'],
        'tnet_lf': ['text_indices', 'aspect_indices', 'aspect_in_text'],
        'aoa': ['text_indices', 'aspect_indices'],
        'mgan': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'bert_spc': ['concat_bert_indices', 'concat_segments_indices'],
        'aen_bert': ['text_bert_indices', 'aspect_bert_indices'],
        'lcf_bert': ['concat_bert_indices', 'concat_segments_indices', 'text_bert_indices', 'aspect_bert_indices'],
    }

    class Option(object):
        pass

    opt = Option()
    opt.model_name = 'bert_spc'
    opt.model_class = model_classes[opt.model_name]
    opt.dataset = 'comment'
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.state_dict_path = 'state_dict/bert_spc_comment_val_acc_0.7636'
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.max_seq_len = 85
    opt.bert_dim = 768
    opt.pretrained_bert_name = 'bert-base-uncased'
    opt.polarities_dim = 3
    opt.hops = 3
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.local_context_focus = 'cdm'
    opt.SRD = 3
    opt.dropout = 0.1

    # Create an Inferer instance
    inf = Inferer(opt)

    # Sử dụng câu truyền từ dòng lệnh
    sentence = args.sentence.lower()
    print(f"sentence: {sentence}")

    # Giả sử inf.extract_aspect trả về danh sách các khía cạnh từ câu
    aspects = inf.extract_aspect(sentence)

    if aspects:
        for aspect in aspects:
            print(f"Extracted aspect: {aspect}")
            
            # Evaluate the sentiment cho từng khía cạnh
            t_probs = inf.evaluate(sentence,aspect)
            pred_labels = t_probs.argmax(axis=-1) - 1

            # Kiểm tra kết quả dự đoán
            for i, label in enumerate(pred_labels):
                aspect_index = i + 1  # Bắt đầu đếm từ 1 thay vì 0
                if label == 1:
                    print(f"Prediction for aspect {aspect_index} ('{aspect}'): Positive")
                elif label == -1:
                    print(f"Prediction for aspect {aspect_index} ('{aspect}'): Negative")
                else:
                    print(f"Prediction for aspect {aspect_index} ('{aspect}'): Neutral")

    else:
        print("No aspect extracted")
