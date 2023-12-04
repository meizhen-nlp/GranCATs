from math import fabs
from multiprocessing.spawn import prepare
import os
import sys
import logging
import random
from time import time
from turtle import forward
import xxlimited
from tqdm import tqdm, trange
import torch 
import torch.nn as nn
import torch.nn.functional as F
from info_nce import InfoNCE
import torch.utils.data as Data
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import transformers.adapters.composition as ac
from transformers import AutoModel, AdapterConfig, AdamW, get_cosine_with_hard_restarts_schedule_with_warmup, BertForMaskedLM, AutoConfig
from accelerate import Accelerator, DistributedDataParallelKwargs
from data import wiki_summary_loader
from utils import save_model, load_model
#import torch.distributed as dist
#dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class summary_adapter(nn.Module):
    def __init__(self, args,dropout=0.1):
        super(summary_adapter, self).__init__()
        # load the pre-trained model———>"bert-base-uncased" or "facebook/bart-base"
        # self.net = BertForMaskedLM.from_pretrained(args.model_dir,output_hidden_states=True)
        self.net = AutoModel.from_pretrained(args.model_dir)
        self.training = True
        self.do_mlm = False
        self.hidden_embedding = args.hidden_embedding
        self.config = AutoConfig.from_pretrained(args.model_dir)
        
        """
        if self.training:
            # adapters
            config_ada = AdapterConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu", 
                                    original_ln_before=False, original_ln_after=True, 
                                    ln_before=False, ln_after=False, 
                                    residual_before_ln=False, adapter_residual_before_ln=False)   
            self.net.add_adapter("label", config=config_ada)
            self.net.set_active_adapters("label")
            # self.net.add_adapter("sentence", config=config_ada)
            # self.net.add_adapter("summary", config=config_ada)
            # self.net.add_adapter_fusion(["label","sentence","summary"])
            # self.net.activate_adapters = ac.Fuse("label","sentence","summary") 

        # Thself.domain_classifier.add_module('fc1', nn.Linear(self.hidden_embedding, int(self.hidden_embedding / 2)))e adversirial loss on domain class
        """
    def forward(self, **inputs):
        features = self.net(**inputs)['last_hidden_state']
        
        return features


def loss_constractive(args, outputs_en, outputs_xx, l2=False):
    # deal with the long paragraph with slipping window, and don't concat label embedding
    outputs_en.to(args.device)
    outputs_xx.to(args.device)
    # in_batch negative sampling
    output_en = torch.mean(outputs_en, dim=1)
    output_xx = torch.mean(outputs_xx, dim=1)
    
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # Contractive Learning
    '''
    Can be used without explicit negative keys whereby each sample is compared
    with the other samples in the batch.
    # pip install info-nce-pytorch
    '''
    loss = InfoNCE()
    loss_con = loss(output_en, output_xx)
    xx_index = []
    '''
    for i in range(output_xx.shape[0]):
        xx_index.append(i)
    random.shuffle(xx_index)
    output_xx_neg = output_xx[xx_index]
    # l2-norm loss
    '''
    loss_l2 = 0
    loss_con_l2 = nn.MSELoss()

    if l2 == True:
        loss_l2 = loss_con_l2(output_en, output_xx) / (loss_con_l2(output_en, output_xx) + loss_con_l2(output_en, output_xx_neg))
        return loss_con +  loss_l2
    else:
        return loss_con

def loss_cat(args, input_1, input_2, list_1, list_2, l2=False):
    lab_en = torch.zeros((args.batch_size, 1, args.hidden_embedding))
    lab_xx = torch.zeros((args.batch_size, 1, args.hidden_embedding))
    sum_en = torch.zeros((args.batch_size, 1, args.hidden_embedding))
    sum_xx = torch.zeros((args.batch_size, 1, args.hidden_embedding))
    for row, ind in enumerate(list_1):
        lab_en_row = torch.mean(input_1[row,1:ind,:].unsqueeze(0), dim=1) # size is (1,768) and the shape of input_1 is (16,351,768)
        sum_en_row = torch.mean(input_1[row,ind+1:,:].unsqueeze(0), dim=1)
        lab_en[row] = lab_en_row.unsqueeze(0) # the shape is (16,1,768)
        sum_en[row] = sum_en_row.unsqueeze(0) # the shape is (16,1,768)
    # print('========================> the shape of entity is {}'.format(lab_en.shape))
    
    for row, ind in enumerate(list_2):
        lab_xx_row = torch.mean( input_2[row,1:ind,:].unsqueeze(0), dim=1) # size is (1,1,768)
        sum_xx_row = torch.mean( input_2[row,ind+1:,:].unsqueeze(0), dim=1)
        lab_xx[row] = lab_xx_row.unsqueeze(0) # the shape is (16,1,768)
        sum_xx[row] = sum_xx_row.unsqueeze(0) # the shape is (16,1,768)      
    
    cat_en = torch.cat((lab_en,sum_en), dim=1)
    cat_xx = torch.cat((lab_xx,sum_xx), dim=1)
    sum_en = torch.mean(cat_en, dim=1)
    sum_xx = torch.mean(cat_xx, dim=1)
    '''
    lab_en = lab_en.squeeze()
    lab_xx = lab_xx.squeeze()
    sum_en = sum_en.squeeze()
    sum_xx = sum_xx.squeeze()
    
    lab_en.to(args.device)
    lab_xx.to(args.device)
    '''
    sum_en.to(args.device)
    sum_xx.to(args.device)
    

    loss = InfoNCE()  
    # loss_lab = loss(lab_en, lab_xx)
    loss_sry = loss(sum_en, sum_xx)
    
    xx_index = []
    for i in range(sum_xx.shape[0]):
        xx_index.append(i)
    random.shuffle(xx_index)
    sum_xx_neg = sum_xx[xx_index]
    
    # l2-nor
    loss_l2 = 0
    loss_con_l2 = nn.MSELoss()
    if l2 == True:
        loss_l2 = loss_con_l2(sum_en, sum_xx) / (loss_con_l2(sum_en, sum_xx) + loss_con_l2(sum_en, sum_xx_neg))
        return loss_lab + loss_sry +  loss_l2
    else:
        # return loss_lab, loss_sry, loss_lab + loss_sry
        return loss_sry


def train_adapters(args, model_adapter):
    # load data, model and optimizer
    
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    summary_dataset = wiki_summary_loader(args)
    train_sampler = Data.RandomSampler(summary_dataset) if args.local_rank == -1 else DistributedSampler(summary_dataset)
    summary_data = Data.DataLoader(dataset=summary_dataset, sampler = train_sampler, batch_size=1, num_workers=1)
    optimizer = AdamW(model_adapter.parameters(), lr=args.lr, eps=args.adam_epsilon)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_cycles=10,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=len(summary_data))

    model_adapter.to(args.device)

    # Before fp16,the model should be convey to GPU
    if args.fp16:
        from apex import amp
        model_adapter, optimizer = amp.initialize(model_adapter, optimizer, opt_level = "O1") # "O1": That means the most of parameters are half-float.

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model_adapter = torch.nn.DataParallel(model_adapter)
    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model_adapter = torch.nn.parallel.DistributedDataParallel(
            model_adapter, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # training
    # summary: en-xx
    time_start = time()
    num_step = 0
    loss_sum = 0.0
    loss_lab = 0.0
    total_loss_first = 0.0
    total_loss = 0.0
    total_loss1 = 0.0
    total_loss2 = 0.0
    set_seed(args)
    logging_loss = 0.0
    global_step = 1
    loss_MLM = torch.nn.CrossEntropyLoss()

    for iter in trange(args.num_epoch):
        print("====> Fine-tuning: training the full model | Iteration is {} <====".format(iter))
        
        for step,item_data in enumerate(summary_data):  # a batch_size
            # which strategy should be adopted to process the input
            input_lab_en, input_lab_xx = item_data
            
            # this is for MLM training
            # input_lab_en, input_lab_en_mask, labels_en, input_lab_xx, input_lab_xx_mask, labels_xx, masked_indices_en, masked_indices_xx = item_data
            optimizer.zero_grad()

            input_lab_en = {k:torch.squeeze(v).to(args.device) for k, v in input_lab_en.items()}
            input_lab_xx = {k:torch.squeeze(v).to(args.device) for k, v in input_lab_xx.items()}
            
            for name, param in model_adapter.net.named_parameters():
                param.requires_grad = True
            with torch.no_grad(): # clc-head is closed for training
                # model_adapter.module.net.requires_grad_(False)
                model_adapter.net.requires_grad_(False)
            for name, param in model_adapter.net.named_parameters():
                if "summary" in name:
                    param.requires_grad = True
                
            # to calculate the loss of CL
            feature_en = model_adapter(**input_lab_en)
            feature_xx = model_adapter(**input_lab_xx)
            loss_cl = loss_constractive(args,feature_en, feature_xx)
            loss_item = loss_cl.item()
            total_loss += loss_item
            
            """
            to calculate the loss of MLM
            """
            # masked_indices_en = masked_indices_en.squeeze().to(args.device)
            # masked_indices_xx = masked_indices_xx.squeeze().to(args.device)
            # loss_en = loss_MLM(logits_en[masked_indices_en].view(-1, model_adapter.config.vocab_size), labels_en[masked_indices_en].view(-1))
            # loss_xx = loss_MLM(logits_xx[masked_indices_xx].view(-1, model_adapter.config.vocab_size), labels_xx[masked_indices_xx].view(-1))
            # loss_back = torch.add(loss_en, loss_xx)
            # loss_item = loss_en.item() + loss_xx.item()


            num_step += 1

            if args.n_gpu > 1:
                loss_cl = loss_cl.mean()  # mean() to average on multi-gpu parallel training

            if args.accumulation_steps > 1:
                loss_cl = loss_cl / args.accumulation_steps

            loss_cl.backward()

            if ((step+1) % args.accumulation_steps) == 0:
                # upgrade the parameters
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model_adapter.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1
                
            # finish the training process of one batch_size
            if torch.distributed.get_rank() == 0 and num_step % 1e2 == 0:
                time_span = round(time() - time_start, 2)
                loss_avg_lb = round(total_loss / num_step, 4)
                print("num_step: ", num_step, "| time ====> ", str(round(time_span/60, 4)) +"mins", "| fine-tuning loss on CL for summary ===> ", loss_avg_lb)
                
        summary_dataset = wiki_summary_loader(args)
        train_sampler = Data.RandomSampler(summary_dataset) if args.local_rank == -1 else DistributedSampler(summary_dataset)
        summary_data = Data.DataLoader(dataset=summary_dataset, sampler = train_sampler, batch_size=1, num_workers=1)
    if not os.path.exists(args.tmp_dir):
        os.makedirs(args.tmp_dir)
    # just save the model stored in rank0 process

    model_to_save = model_adapter.module if hasattr(model_adapter, "module") else model_adapter
    torch.save(model_to_save.state_dict(), os.path.join(args.tmp_dir, "pytorch_model.bin"))
    
    """
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        model_to_save = model_adapter.module if hasattr(model_adapter, "module") else model_adapter
        save_model(model_to_save, args.tmp_dir, False)
    del model_adapter
    if args.local_rank in [-1, 0]:
        tb_writer.close()

    """
    return

def multi_retriever(args):
    # train adapter 
    # Load pretrained model and tokenizer
    
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
    model_adapter = summary_adapter(args)
    if args.local_rank == 0:
        torch.distributed.barrier()
    train_adapters(args, model_adapter)
    
    for i in range(1,7):
        print('=================================> pretrain the full model with CL {} iteration <================================'.format(i))
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()
        model_adapter = summary_adapter(args)
        if args.local_rank == 0:
            torch.distributed.barrier()
        # load_model(model_adapter, args.tmp_dir)
        model_adapter.load_state_dict(torch.load(os.path.join(args.tmp_dir, "pytorch_model.bin"), map_location='cpu'))
        train_adapters(args, model_adapter)
     
        
        
    
    

    

            
            
        


    

    
