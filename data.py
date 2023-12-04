'''
(1) Download the parallel summary dataset to constitute the batch_size
{id:   , labels: {en:   , de:   ,  }}, summary:{en:  ,   de:   }}   210w 
'''

import os
import re
import json
import sys
import logging
import random
import collections
import numpy as np
from tqdm import tqdm
import transformers
import torch
import torch.utils.data as Data 
from transformers import AutoTokenizer, DataProcessor
from torch.utils.data import DataLoader
logger = logging.getLogger(__name__)
class wiki_summary_loader(Data.Dataset):
    def __init__(self, args, hard_neg_sampling=False, shuffle=True):
        # load the parallel summary
        num_id = 0
        list_id = []  
        summary_dict = {}
        sentence_dict = {}
        labels_dict = {}
        # with open(os.path.join(args.data_file, 'test.json'), 'r+', encoding='utf-8') as reader:
        with open(os.path.join(args.data_file, 'new_add_sentence_5.json'), 'r+', encoding='utf-8') as reader:
        # with open(os.path.join(args.data_file, 'test_384.json'), 'r+', encoding='utf-8') as reader:
            for line in tqdm(reader, desc=" process the summary_data"): 
                line = json.loads(line)
                list_id.append(line["id"])
                if line["id"] not in summary_dict.keys():
                    num_id += 1
                    summary_dict[line["id"]] = []
                    # delete the special syntactic
                    del_summary = self._rpl_whitespace(line["summary"]["en"].strip())
                    summary_dict[line["id"]].append(del_summary)
                for k, v in line["summary"].items():
                    if k != "en":
                        v = self._rpl_whitespace(v.strip())
                        summary_dict[line["id"]].append(v)        
                if line["id"] not in labels_dict.keys():
                    labels_dict[line["id"]] = []
                    labels_dict[line["id"]].append(line["labels"]["en"].strip())
                for k, v in line["labels"].items():
                    if k != "en":
                        labels_dict[line["id"]].append(v.strip())
                if line["id"] not in sentence_dict.keys():
                    sentence_dict[line["id"]] = []
                    # delete the special syntactic
                    del_sentence = self._rpl_whitespace(line["sentence"]["en"].strip())
                    sentence_dict[line["id"]].append(del_sentence)
                for k, v in line["sentence"].items():
                    if k != "en":
                        v = self._rpl_whitespace(v.strip())
                        sentence_dict[line["id"]].append(v)
        if shuffle:
            random.shuffle(list_id)
       
        self.list_id = (i for i in list_id) # convert to a generator      
        self.num_id = num_id
        self.batch_size = args.batch_size
        self.summary_dict = summary_dict
        self.sentence_dict = sentence_dict
        self.max_length = args.max_length
        self.labels_dict = labels_dict
        self.summary_pool = set(summary_dict.keys())
        self.labels_pool = set(labels_dict.keys())  # the return type is dict
        # judgement of processing summary data
        self.no_sliding = args.no_sliding
        self.do_mlm = args.do_mlm
        self.w_label = args.w_label
        # set tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.lm_mask_token = self.tokenizer.mask_token
        self.lm_mask_token_id = self.tokenizer.mask_token_id

    def _rpl_whitespace(self, str):
        str = re.sub(r"[\r\t\n]"," ",str)
        return str

    def negative_sampler(self, neg_num):
        # global negative samples on summary
        neg_sum_id = random.sample(self.summary_pool, neg_num)  # the type is list through sampling
        for id in neg_sum_id:
            neg_sum_sample = random.choice(self.summary_dict[id][1:])
        # global negative samples on entity
        neg_lab_id = random.sample(self.labels_pool, neg_num)
        for id in neg_lab_id:
            neg_lab_sample = random.choice(self.labels_dict[id][1:])
        return neg_sum_sample, neg_lab_sample

    def mask_tokens(self, inputs: torch.Tensor):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, 0.15)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels, masked_indices


               
    def __len__(self):
        return self.num_id
   
    def sample_frequency(List:list):
        frequecy_dict = {}
 
    def __getitem__(self, index):
        xx1_summary = [] 
        xx2_summary = []
        xx1_sentence = []
        xx2_sentence = []
        en_first = []
        xx_first = []
        xx1_label = []
        xx2_label = [] 
        for i in range(self.batch_size): 
            id = self.list_id.__next__()
              
            xx1_summary.append(self.summary_dict[id][0])  # length = batch_size, summary_en
            xx2_summary.append(random.choice(self.summary_dict[id][1:])) # length = batch_size, summary_xx, type is list []
            # en_first.append((self.summary_dict[id][0]).split('.')[0])
            # xx_first.append(random.choice(self.summary_dict[id][1:]).split('.')[0])
            xx1_sentence.append(self.sentence_dict[id][0])
            xx2_sentence.append(random.choice(self.sentence_dict[id][1:]))
            
            xx1_label.append(self.labels_dict[id][0])
            xx2_label.append(random.choice(self.labels_dict[id][1:]))
            """
            summary = np.random.choice(self.summary_dict[id][0:], 2)
            xx1_summary.append(summary[0])
            xx2_summary.append(summary[1])
            sentence = np.random.choice(self.sentence_dict[id][0:], 2)
            xx1_sentence.append(sentence[0])
            xx2_sentence.append(sentence[1])
            label = np.random.choice(self.labels_dict[id][0:], 2)
            xx1_label.append(label[0])
            xx2_label.append(label[1])
            """
        return self.input_process(xx1_summary, xx2_summary, xx1_sentence, xx2_sentence, xx1_label, xx2_label)
    
    def get_attention(self, input_dic):
        inputs_masked, labels, masked_indices = self.mask_tokens(input_dic["input_ids"])
        attention_mask = []
        for input_ids in inputs_masked:
          att_mask = [int(token_id != self.tokenizer.pad_token_id) for token_id in input_ids]
          att_mask = [int(token_id != self.tokenizer.mask_token_id) for token_id in input_ids]
          attention_mask.append(att_mask)
        input_dic["input_ids"] = torch.tensor(inputs_masked)
        input_dic["attention_mask"] = torch.tensor(attention_mask)
        # input_dic["labels"] = labels
        return input_dic, labels, masked_indices

    def input_process(self, xx1_summary, xx2_summary, xx1_sentence, xx2_sentence, xx1_label, xx2_label):
        
        # input_sum_en = self.tokenizer(xx1_summary , padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        # input_sum_xx = self.tokenizer(xx2_summary,  padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        
        # input_first_en = self.tokenizer(xx1_sentence, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        # input_first_xx = self.tokenizer(xx2_sentence,  padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        
        # input_lab_en_mask, labels_en, masked_indices_en = self.get_attention(input_lab_en)
        # input_lab_xx_mask, labels_xx, masked_indices_xx = self.get_attention(input_lab_xx)
        input_lab_en = self.tokenizer(xx1_summary,  padding=True, truncation=True, max_length=384, return_tensors="pt")
        input_lab_xx = self.tokenizer(xx2_summary,  padding=True, truncation=True, max_length=384, return_tensors="pt") 
        
             
        return  input_lab_en,  input_lab_xx
            
            
            
                   
              
        
        





