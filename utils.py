import os
import json
import random
import torch
import torch.utils.data as Data
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate import Accelerator, DistributedDataParallelKwargs
from transformers.adapters.composition import Fuse
seed = 123
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# def save_model(model, accelerator, path, fusion=False):
def save_model(model, path, fusion=False):
    # model = accelerator.unwrap_model(model)
    if fusion:
        model.net.save_adapter_fusion(path, "label,sentence,summary")
    else:
        model.net.save_all_adapters(path)
    return

def load_model(model, path, fusion=False):
    # model.net.load_adapter(os.path.join(path, "label"))
    # model.net.load_adapter(os.path.join(path, "sentence"))
    # model.net.load_adapter(os.path.join(path, "summary"))

    if fusion:
        # model.net.load_adapter_fusion(os.path.join(path, "label, sentence, summary"))
        model.net.add_adapter_fusion(Fuse("label", "sentence", "summary"))
        model.net.set_active_adapters(Fuse("label", "sentence", "summary"))    
    return
