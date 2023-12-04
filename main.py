import argparse
import torch
from model import multi_retriever
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the multilingual retriever")
    
    # data_file
    parser.add_argument("--data_file", type=str, default="/home/guoxu/meizhen/xtreme/code_mbert/", help="the input data directory")
    parser.add_argument("--model_dir", type=str, default="/home/guoxu/meizhen/xtreme/bert-base-multilingual-cased", help="the Language Model directory")
    parser.add_argument("--tmp_dir", type=str, default= "/home/guoxu/meizhen/xtreme/code_mbert_ft/mbert_cl_ft_sm/", help="The stored model directory.")
    parser.add_argument("--tmp_dir_fusion", type=str, default="/cluster/home/meiliu/scratch/adapter_fusion", help="The stored fusion model directory.")
        
    # model parameter
    parser.add_argument("--discriminator", type=str, default=False, help="whether to adapt the discriminator to classify the language label")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on on gpus")
    parser.add_argument("--fp16", action="store_true",help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--seed", type=int, default=123, help="random seed for initialization")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type selected in the list [mbert, bart, roberta-base, roberta-large]")
    parser.add_argument("--w_label", type=bool, default=False, help=" with the labels(entity) to align the different language semantics")
    parser.add_argument("--do_mlm", type=bool, default=False, help=" whether to add the loss of masked language model to train this model")
    parser.add_argument("--batch_size", type=int, default=8, help="the specific number of example every iter")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="the accumulated steps for erreos")
    parser.add_argument("--num_epoch", type=int, default=1, help="the total number of training iteration")
    parser.add_argument("--num_epoch_fusion", type=int, default=2, help="the total number of fusion training iteration")
    parser.add_argument("--max_length", type=int, default=384, help="the maximum length od input of Bert ")
    parser.add_argument("--hidden_embedding", type=int, default=768, help="768 for mbert")
    parser.add_argument("--no_sliding", type=bool, default=False, help=" to determine whether to use the sliding window or not, otherwise truncation")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--fusion", type=bool, default=False, help=" to fuse the summary and label adapters")
    parser.add_argument("--device", type=int, default=0, help="which GPU to use. set -1 to use CPU.")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate of GCS.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--warmup_steps", type=int, default=1e4, help="number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay value")
    
    args = parser.parse_args()
     # Setup CUDA, GPU & distributed training
    if args.local_rank == -1:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      torch.distributed.init_process_group(backend="nccl")
      args.n_gpu = 1
    
    if args.n_gpu == 1:
      dist.init_process_group('gloo', init_method='file:///tmp/somefile', rank=0, world_size=1)
    
    args.device = device
    print(args)
    print(torch.cuda.is_available())
    multi_retriever(args)
    
