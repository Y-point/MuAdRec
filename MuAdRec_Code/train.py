import os
import time
import torch
import random
import numpy as np
import pandas as pd
import argparse
import logging
import pickle
import sys
from datetime import datetime
from torch import nn
import torch.nn.functional as F
try:
    import yaml
except ImportError:
    yaml = None

from torch.utils.data import Dataset, DataLoader

from muadrec import MuAdRec
from utils import evaluate

class SeqDataset(Dataset):
    def __init__(self, data):
        self.seq_data = [torch.tensor(seq, dtype=torch.long) for seq in data['seq']]
        self.len_seq_data = [torch.tensor(len_seq, dtype=torch.long) for len_seq in data['len_seq']]
        self.next_data = [torch.tensor(next_val, dtype=torch.long) for next_val in data['next']]

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, idx):
        return {'seq': self.seq_data[idx], 'len_seq': self.len_seq_data[idx], 'next': self.next_data[idx]}

def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

logging.getLogger().setLevel(logging.INFO)

def setup_seed(seed): 
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

def parse_config_files(config_files):
    if not config_files:
        return {}
    if yaml is None:
        raise ImportError("PyYAML is required for --config_files. Install with: pip install pyyaml")
    merged = {}
    paths = [x.strip() for x in config_files.split(",") if x.strip()]
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError(f"Config file must contain a YAML mapping: {p}")
        merged.update(cfg)
    return merged

def build_parser():
    parser = argparse.ArgumentParser(description="Run AlphaFuseMMComplete.")
    parser.add_argument('--config_files', type=str, default=None,
                        help='Comma-separated YAML config files.')
    parser.add_argument('--random_seed', type=int, default=22)
    ### training settings
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--lr_delay_rate', type=float, default=0.99)
    parser.add_argument('--lr_delay_epoch', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=500,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='ASO',
                        help='Dataset name')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device.')
    parser.add_argument('--l2_decay', type=float, default=1e-6,
                        help='l2 loss reg coef.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    ### SASRec backbone settings
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                        help='dropout ')
    ### loss function parameters
    parser.add_argument('--loss_type', type=str, default="infoNCE")
    parser.add_argument('--neg_ratio', type=int, default=64,
                        help='#Negative:#Positive = neg_ratio.')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='tao.')
    ### language embeddings settings
    parser.add_argument('--language_model_type', default="3large", type=str)
    parser.add_argument('--language_embs_scale', default=40, type=int)

    ### ID embeddings settings
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Number of hidden factors, i.e., ID embedding size.')
    parser.add_argument('--ID_embs_init_type', type=str, default="normal")
    ### model selection
    parser.add_argument('--model_type', type=str, default="MuAdRec")
    # AlphaFuse
    parser.add_argument('--null_thres', type=float, default=None,)
    parser.add_argument('--null_dim', type=int, default=128,)
    parser.add_argument('--standardization', type=str2bool, default=True)
    parser.add_argument('--cover', type=str2bool, default=False)
    parser.add_argument('--ID_space', type=str, default="singular")
    parser.add_argument('--inject_space', type=str, default="singular")
    # AlphaFuse innovations
    parser.add_argument('--adaptive_null_dim', type=str2bool, default=True)
    parser.add_argument('--adaptive_null_min_dim', type=int, default=32)
    parser.add_argument('--adaptive_null_gamma', type=float, default=1.0)
    parser.add_argument('--adaptive_null_strategy', type=str, default="inv_pop")
    parser.add_argument('--hier_inject', type=str2bool, default=True)
    parser.add_argument('--hier_inject_pre_attn', type=str2bool, default=True)
    parser.add_argument('--hier_inject_post_attn', type=str2bool, default=True)
    parser.add_argument('--hier_inject_use_gate', type=str2bool, default=True)
    # Multi-modal AlphaFuse innovations
    parser.add_argument('--use_shared_null_inject', type=str2bool, default=True)
    parser.add_argument('--use_cross_modal_null_align', type=str2bool, default=True)
    parser.add_argument('--modal_fuse_alpha', type=float, default=0.6)
    parser.add_argument('--align_temperature', type=float, default=0.3)
    parser.add_argument('--image_embs_scale', type=float, default=40)
    parser.add_argument('--beta_mm', type=float, default=0.2,
                        help='scale of cross-modal null alignment loss')
    parser.add_argument('--log_file', type=str, default=None,
                        help='Optional log file path. Auto-generated if omitted.')
    return parser

def parse_args():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config_files', type=str, default=None)
    pre_args, remaining = pre_parser.parse_known_args()

    parser = build_parser()
    config_defaults = parse_config_files(pre_args.config_files)
    if config_defaults:
        parser.set_defaults(**config_defaults)
    args = parser.parse_args(remaining)
    if pre_args.config_files is not None:
        args.config_files = pre_args.config_files
    return args

def setup_logging(args):
    log_dir = './log'
    os.makedirs(log_dir, exist_ok=True)
    log_file = getattr(args, "log_file", None)
    use_auto_log = not isinstance(log_file, str) or not log_file.strip()
    if use_auto_log:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(log_dir, f"{args.data}_MuAdRec_{ts}.log")
    else:
        args.log_file = log_file.strip()
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    log_f = open(args.log_file, "a", encoding="utf-8")
    sys.stdout = TeeStream(sys.stdout, log_f)
    sys.stderr = TeeStream(sys.stderr, log_f)
    print(f"[Log] writing to: {args.log_file}")
    return log_f

if __name__ == '__main__':

    args = parse_args()
    log_handle = setup_logging(args)
    setup_seed(args.random_seed)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    key_words = vars(args)
    data_directory = './' + args.data
    model_directory = './saved/' + args.data
    os.makedirs(model_directory, exist_ok=True)
    
    key_words["language_embs_path"] = data_directory

    model = MuAdRec(device, **key_words).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params}")
    print(f"Trainable Parameters: {trainable_params}")
    
    print(key_words)

    train_data = pd.read_pickle(os.path.join(data_directory, 'train_data.df'))
    train_data.reset_index(inplace=True,drop=True)
    train_dataset = SeqDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)

    val_data = pd.read_pickle(os.path.join(data_directory, 'val_data.df'))
    val_data.reset_index(inplace=True,drop=True)
    val_dataset = SeqDataset(val_data)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    test_data = pd.read_pickle(os.path.join(data_directory, 'test_data.df'))
    test_data.reset_index(inplace=True,drop=True)
    test_dataset = SeqDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    best_ndcg20 = 0
    patience = 100
    counter = 0
    step = 0
    T = 0.0
    
    print("Loading Data Done.")
    t0 = time.time()
    val_ndcg20 = evaluate(model, val_loader, device)
    t1 = time.time() - t0
    print("\n using ", t1, "s ", "Eval Time Cost", T, "s.")
    
    for epoch in range(args.epoch):
        model.train()
        for batch in train_loader:
            batch_size = len(batch['seq'])
            seq = batch['seq'].to(device)
            target = batch['next'].to(device)
            
            optimizer.zero_grad()
            
            if args.loss_type == "CE":
                loss = model.calculate_ce_loss(seq, target)
            elif args.loss_type == "BCE":
                loss = model.calculate_bce_loss(seq, target, args.neg_ratio)
            elif args.loss_type == "infoNCE":
                loss = model.calculate_infonce_loss(seq, target, args.neg_ratio, args.temperature)
            
            if args.use_cross_modal_null_align:
                mm_align_loss = model.cross_modal_null_align_loss(target)
                loss = loss + args.beta_mm * mm_align_loss

            loss.backward()
            optimizer.step()
            step += 1
            
        print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item()))
        
        if (epoch + 1) % 50 == 0:
            _ = evaluate(model, train_loader, device)
            
        if (epoch + 1) % 1 == 0:
            model.eval()
            print('-------------------------- EVALUATE PHRASE --------------------------')
            t0 = time.time()
            val_ndcg20 = evaluate(model, val_loader, device)
            t1 = time.time() - t0
            print("\n using ", t1, "s ", "Eval Time Cost", T, "s.")

            model.train()
            tv_ndcg20 = val_ndcg20 
            
            if tv_ndcg20 > best_ndcg20:
                best_ndcg20 = tv_ndcg20
                counter = 0
                print("\n best NDCG@20 is updated to ", best_ndcg20, "at epoch", epoch)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                epoch_str = f"MuAdRec_rs{args.random_seed}_IDdim{args.hidden_dim}_Textdim{args.null_dim}_{args.lr}_{args.loss_type}_{ts}.pth"
                torch.save(model.state_dict(), os.path.join(model_directory, epoch_str))
            else:
                counter += 1
                if counter >= patience:
                    break
            print('----------------------------------------------------------------')
            
    model.load_state_dict(torch.load(os.path.join(model_directory, epoch_str)))
    items_emb = model.return_item_emb()
    
    model.eval()
    print('-------------------------- TEST RESULTS --------------------------')
    _ = evaluate(model, test_loader, device)
    print("Done.")
    log_handle.close()
