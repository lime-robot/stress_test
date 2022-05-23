import os
import math
import torch
import time
import glob
import copy
import random
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import vpp_data
import vpp_model
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, KFold
from torch.cuda.amp import autocast, GradScaler
from transformers import (AdamW, get_linear_schedule_with_warmup, 
                          get_constant_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup)
from transformers.models.auto.tokenization_auto import AutoTokenizer

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import RandomSampler, SequentialSampler

#import wandb
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_colwidth', 100)

warnings.filterwarnings(action='ignore')

#os.environ["NCCL_SOCKET_IFNAME"] = "enp69s0"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

VERSION=os.path.abspath(__file__).split('/')[-2]
MODEL_DIR=f'../model/{VERSION}'
SUBMISSION_DIR=f'../submission/{VERSION}'


def main():    
    parser = argparse.ArgumentParser(description="Ventilator Pressure Prediction")
    parser.add_argument("--batch_size", type=int, default=64,            
            help="input batch size for training (default: 64)")    
    parser.add_argument("--num_workers", type=int, default=4,   
            help="number of workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--train_file", type=str, default='train.zip',
                        help="the name of the training file (default: train.zip")
    parser.add_argument("--test_file", type=str, default='test.zip',
                        help="the name of the test file (default: test.zip")
    parser.add_argument("--extra_file", type=str, default='') # papers.csv                        
    parser.add_argument("--input_dir", type=str, default='../input',
                        help="input directory (default: ../input")    
    parser.add_argument("--k", type=int, default=5,
                        help="choose k in k-fold. k is the number of  data set groups (default: 5)")    
    parser.add_argument("--model_dir", type=str, default='model',
            help="specify the pre-trained model directory (default: '')")
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--valid_freq", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-04,  
        help="learning rate (default: 1e-04)")
    parser.add_argument("--dropout", type=float, default=0.1,  
        help="dropout rate (default: 0.2)")
    parser.add_argument("--weight_decay", type=float, default=0.01,  
        help="weight_decay (default: 0.01)")
    parser.add_argument("--warmup_steps", type=float, default=20,  
        help="""increase the additional weight of learning_rate linearly """
                   """from 0 to 1 during warmup_steps (default: 20)""")
    parser.add_argument("--arch",  type=str, default='transformer',
                        help="the name of the model architecture (default: transformer')")
    parser.add_argument("--n_folds", type=int, default=12)
    parser.add_argument("--fold", type=int, default=9,
                    help="specify i-th fold to train (default: 9)")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--max_seq_len", type=int, default=80,
                    help="maximum length of excerpt text (default: 80)")
    parser.add_argument("--ckpt_path", type=str, default='',
                    help="path of the model checkpoint  (default: '')")
    parser.add_argument("--feature_size", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--target_size", type=int, default=1)        
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--optim", type=str, default='AdamW')
    parser.add_argument("--submission_dir", type=str, default='submissions')
    parser.add_argument("--pseudo_path", type=str, default='')
    parser.add_argument("--scheduler", type=str, default='cosine')
    parser.add_argument("--max_grad_norm", type=float, default=1000)    
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--augmentation", action='store_true')
    parser.add_argument("--wsteps", type=int, default=15)
    parser.add_argument("--extra", type=str, default='cosine')
    args = parser.parse_args()
    
    # Set a fixed random seed to reproduce the same result when the code is executed 
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)    
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True  
    
    os.makedirs(os.path.join(args.submission_dir, 'verify'), exist_ok=True)  
         
    train_df = pd.read_csv(os.path.join(args.input_dir, args.train_file))
    test_df = pd.read_csv(os.path.join(args.input_dir, args.test_file))
    
    train_df['step'] = list(range(80))*train_df['breath_id'].nunique()
    test_df['step'] = list(range(80))*test_df['breath_id'].nunique()
    
    train_df['pressure_diff1'] = train_df['pressure'].diff(periods=1)
    train_df['pressure_diff2'] = train_df['pressure'].diff(periods=2)
    train_df['pressure_diff3'] = train_df['pressure'].diff(periods=3)
    train_df['pressure_diff4'] = train_df['pressure'].diff(periods=4)
    train_df.loc[train_df['step']<1, 'pressure_diff1'] = 0 #  faster than groupby operation. 
    train_df.loc[train_df['step']<2, 'pressure_diff2'] = 0
    train_df.loc[train_df['step']<3, 'pressure_diff3'] = 0
    train_df.loc[train_df['step']<4, 'pressure_diff4'] = 0    
    
    train_df['u_in_diff'] = train_df['u_in'].diff()
    train_df.loc[train_df['step']<1, 'u_in_diff'] = 0
    test_df['u_in_diff'] = test_df['u_in'].diff()
    test_df.loc[test_df['step']<1, 'u_in_diff'] = 0
    
    train_df['time_diff'] = train_df['time_step'].diff()
    train_df.loc[train_df['step']<1, 'time_diff'] = 0
    test_df['time_diff'] = test_df['time_step'].diff()
    test_df.loc[test_df['step']<1, 'time_diff'] = 0
    
    train_df['inhaled_air'] = train_df['time_diff']*train_df['u_in']
    test_df['inhaled_air'] = test_df['time_diff']*test_df['u_in']
        
    train_df['RC'] = train_df['R'].astype(str) + '_' + train_df['C'].astype(str)
    test_df['RC'] = test_df['R'].astype(str) + '_' + test_df['C'].astype(str)
    
    rc2idx = dict( [(v, i) for i, v in enumerate(train_df['RC'].astype('category').cat.categories )] )
    train_df['rc_index'] = train_df['RC'].map(rc2idx)
    test_df['rc_index'] = test_df['RC'].map(rc2idx)
    
    r2idx = dict( [(v, i) for i, v in enumerate(train_df['R'].astype('category').cat.categories )] )
    train_df['r_index'] = train_df['R'].map(r2idx)    
    test_df['r_index'] = test_df['R'].map(r2idx)
    
    c2idx = dict( [(v, i) for i, v in enumerate(train_df['C'].astype('category').cat.categories )] )
    train_df['c_index'] = train_df['C'].map(c2idx)
    test_df['c_index'] = test_df['C'].map(c2idx)
         
    train_df['u_in_min'] = train_df['breath_id'].map(dict(train_df.groupby('breath_id')['u_in'].min()))      
    test_df['u_in_min'] = test_df['breath_id'].map(dict(test_df.groupby('breath_id')['u_in'].min()))
    
    args.cont_cols = [
        'u_in', 'u_out',
        'u_in_min',
        'u_in_diff',
        'time_diff',
        'inhaled_air',
       ]
    
    ### Get CV
    meta_df = train_df.groupby('breath_id')[['breath_id', 'r_index', 'c_index']].head(1).reset_index(drop=True)
    meta_df['typ_r_c'] = meta_df['r_index'].astype(str) + '_' + meta_df['c_index'].astype(str)
    kf = StratifiedKFold(args.n_folds, random_state=42, shuffle=True)
    meta_df['fold'] = -1
    for fold, (_, val_idx) in enumerate(kf.split(meta_df, y=meta_df['typ_r_c'])):
        meta_df.loc[val_idx, 'fold'] = int(fold)        
    
    #valid_df = train_df[train_df['breath_id'].isin(train_breath_ids[valid_ids])].reset_index(drop=True)
    #train_df = train_df[train_df['breath_id'].isin(train_breath_ids[train_ids])].reset_index(drop=True)
    val_breath_ids = meta_df.query(f'fold=={args.fold}')['breath_id']
    valid_df = train_df[train_df['breath_id'].isin(val_breath_ids)].reset_index(drop=True)
    train_df = train_df[~train_df['breath_id'].isin(val_breath_ids)].reset_index(drop=True)
    
    if args.pseudo_path != '':
        pseudo_df = pd.read_csv(args.pseudo_path)
        test_df['pressure'] = pseudo_df['pressure']
        test_df['pressure_diff1'] = test_df['pressure'].diff(periods=1)
        test_df['pressure_diff2'] = test_df['pressure'].diff(periods=2)
        test_df['pressure_diff3'] = test_df['pressure'].diff(periods=3)
        test_df['pressure_diff4'] = test_df['pressure'].diff(periods=4)
        test_df.loc[test_df['step']<1, 'pressure_diff1'] = 0 #  faster than groupby operation. 
        test_df.loc[test_df['step']<2, 'pressure_diff2'] = 0
        test_df.loc[test_df['step']<3, 'pressure_diff3'] = 0
        test_df.loc[test_df['step']<4, 'pressure_diff4'] = 0
        train_df = train_df.append(test_df).reset_index(drop=True)
        
    train_db = vpp_data.VPPDataset(train_df, args.cont_cols, times=100, max_seq_len=args.max_seq_len,
                                    augmentation=True)
    valid_db = vpp_data.VPPDataset(valid_df,  args.cont_cols, max_seq_len=args.max_seq_len)
    test_db = vpp_data.VPPDataset(test_df,  args.cont_cols, max_seq_len=args.max_seq_len)
    
    if args.local_rank !=-1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        args.device = torch.device("cuda", args.local_rank)
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()        
    else:
        args.device = torch.device("cuda")
        args.rank = 0
    
    if args.rank == 0:
        print(args)
        #wandb.init(project='google landmark retrieval 2021')
        #wandb.config.update(args)
        print('train set shape:', train_df.shape)
    
    if args.local_rank !=-1:
        train_sampler = DistributedSampler(train_db, shuffle=True, drop_last=True)
    else:        
        train_sampler = RandomSampler(train_db)
    valid_sampler = SequentialSampler(valid_db)
    test_sampler = SequentialSampler(test_db)
    
    train_loader = DataLoader(
        train_db, batch_size=args.batch_size, sampler=train_sampler, 
        num_workers=args.num_workers)
    valid_loader = DataLoader(
        valid_db, batch_size=args.batch_size, sampler=valid_sampler,
        num_workers=args.num_workers)
    test_loader = DataLoader(
        test_db, batch_size=args.batch_size, sampler=test_sampler,
        num_workers=args.num_workers)
        
    model = vpp_model.SognaModel(args)
        
    if args.ckpt_path != "":        
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')        
        state_dict = checkpoint['state_dict']        
        res = model.load_state_dict(state_dict, strict=True)
        if args.rank == 0:
            print(res)  
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.ckpt_path, checkpoint['epoch']))
        del state_dict, checkpoint
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.rank == 0:
        print('parameters: ', count_parameters(model))
    
    model.to(args.device)
    if args.local_rank != -1  :        
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=False)        
    else:
        model = torch.nn.DataParallel(model)
    
    optimizer_grouped_parameters = model.parameters()    
    if args.optim == 'AdamW':
        optimizer = AdamW(optimizer_grouped_parameters,
                               lr=args.lr,
                               weight_decay=args.weight_decay,                           
                               )
        if args.rank == 0:
            print('use AdamW')
    elif args.optim == 'Adam':
        optimizer = torch.optim.Adam(optimizer_grouped_parameters,
                               lr=args.lr,
                               weight_decay=args.weight_decay,                           
                               )
        if args.rank == 0:
            print('use Adam')
    elif args.optim == 'SGD':
        optimizer = torch.optim.SGD(optimizer_grouped_parameters,
                               lr=args.lr,
                               weight_decay=args.weight_decay,                           
                               )
        if args.rank == 0:
            print('use SGD')
    else:
        print(f'{args.optim} is not supported.')
        assert(0)    
    
    
    if args.scheduler == 'constant':                                            
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.wsteps)    
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1)
    elif args.scheduler == 'linear':
        num_train_optimization_steps = len(train_loader) * (args.max_epochs)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=args.wsteps,
                                                num_training_steps=num_train_optimization_steps,                                                
                                                )
    elif args.scheduler == 'cosine':
        num_training_steps = int(len(train_loader)) * args.max_epochs
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    elif args.scheduler == 'onecycle':
        scheduler  = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader), 
                                                     epochs=args.max_epochs,
                                                     max_lr=args.lr, pct_start=0.001, final_div_factor=1e1)
    else:
        print(f'{args.scheduler} is not supported.')
        assert(0)
    
    def get_lr():    
        return scheduler.get_lr()[0]        
    
    log_df = pd.DataFrame()
    curr_lr = get_lr()
        
    def print_msg(msg):
        if args.rank == 0:
            print(msg)
           
    prev_model_path = ''
    best_loss = 1e04
    best_model = None
        
    for epoch in (range(args.start_epoch, args.max_epochs)):        
        if args.local_rank != -1 :
            train_sampler.set_epoch(epoch)
            
        def get_log_row_df(epoch, lr, train_res, valid_res):
            log_row = {'EPOCH':epoch, 'LR':lr,
                       'TRAIN_LOSS':train_res['loss'], 'TRAIN_MAE':train_res['mae'],
                       'VALID_LOSS':valid_res['loss'], 'VALID_MAE':valid_res['mae'],
            }            
            return pd.DataFrame(log_row, index=[0])
        
        
        train_res = train(train_loader, model, optimizer, epoch, scheduler, args)
        if args.scheduler == 'multistep':
            scheduler.step()
        if (epoch % args.valid_freq==0) or (epoch == (args.max_epochs-1)) :            
            valid_res = valid(valid_loader, model, args)
            curr_lr = get_lr()            
            
            # 모델의 파라미터가 저장될 파일의 이름을 정합니다.            
            arch = 'a' + args.arch.replace("_", "-")
            #last_token = ''
                
            curr_model_name = (f'f{args.fold}_b{args.batch_size}_'
                                f'd{args.dropout}_e{epoch}_s{args.seed}_{arch}_'
                                f'{VERSION}_l{valid_res["mae"]:.4f}.pth')
            
            log_row_df = get_log_row_df(epoch, curr_lr, train_res, valid_res)
            # log_df에 결과가 집계된 한 행을 추가합니다.
            log_df = log_df.append(log_row_df, sort=False)
            if args.rank == 0:
                print(log_df.tail(20)) # log_df의 최신 10개 행만 출력합니다.            
            
            model_dir = os.path.join(MODEL_DIR, args.arch)
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, curr_model_name)            
            
            if valid_res['mae'] > 0.12:
                args.valid_freq = 10
            elif valid_res['mae'] > 0.11:
                args.valid_freq = 5
            
            if best_model is None:
                best_model = model
            if best_loss > valid_res['mae']:
                best_loss = valid_res['mae']
                best_model = copy.deepcopy(model)
                
                if args.rank == 0:                  
                    #print(valid_df.shape)
                    #valid_df['prediction'] = 0
                    #valid_df['check_breath_id'] = 0
                    #valid_df.loc[valid_df['step']<args.max_seq_len, 'prediction'] = list(valid_res['prediction'].reshape(-1))
                    #valid_df.loc[valid_df['step']<args.max_seq_len, 'check_breath_id'] = list(valid_res['breath_id'].reshape(-1))
                    #valid_df.to_csv('submissions/valid.csv', index=False)
                    
                    model_to_save = model.module if hasattr(model, 'module') else model
                    """
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model_to_save.state_dict(),
                        'log': log_df,
                        },
                        model_path,
                     )
                    if os.path.isfile(prev_model_path):
                        os.remove(prev_model_path)
                    """
                    prev_model_path = model_path
        
    valid_res = valid(valid_loader, best_model, args)
    #print(valid_df.shape)
    valid_df['prediction'] = 0
    valid_df['check_breath_id'] = 0
    valid_df.loc[valid_df['step']<args.max_seq_len, 'prediction'] = list(valid_res['prediction'].reshape(-1))
    valid_df.loc[valid_df['step']<args.max_seq_len, 'check_breath_id'] = list(valid_res['breath_id'].reshape(-1))
    if args.rank==0:
        valid_df.to_csv(f'submissions/limerobot_valid_fold{args.fold}_seed{args.seed}.csv', index=False)
    
    # make a submission file using the best model    
    test_res = valid(test_loader, best_model, args)
    #print(test_df.shape)
    test_df['prediction'] = 0
    test_df['check_breath_id'] = 0
    test_df.loc[test_df['step']<args.max_seq_len, 'prediction'] = list(test_res['prediction'].reshape(-1))
    test_df.loc[test_df['step']<args.max_seq_len, 'check_breath_id'] = list(test_res['breath_id'].reshape(-1))
    if args.rank==0:        
        test_df.to_csv(f'submissions/verify/limerobot_submission_fold{args.fold}_seed{args.seed}.csv', index=False)
        test_df['pressure'] = test_df['prediction'] 
        test_df[['id', 'pressure']].to_csv(f'submissions/limerobot_submission_fold{args.fold}_seed{args.seed}.csv', index=False)
    

def train(train_loader, model, optimizer, epoch, scheduler, args):
    """    
    한 에폭 단위로 학습을 시킵니다.

    매개변수
    train_loader: 학습 데이터셋에서 배치(미니배치) 불러옵니다.
    model: 학습될 파라미터를 가진 딥러닝 모델
    optimizer: 파라미터를 업데이트 시키는 역할
    scheduler: learning_rate를 감소시키는 역할
    """
    # AverageMeter는 지금까지 입력 받은 전체 수의 평균 값 반환 용도
    batch_time = AverageMeter()     # 한 배치처리 시간 집계
    data_time = AverageMeter()      # 데이터 로딩 시간 집계
    losses = AverageMeter()            # 손실 값 집계        
    maes = AverageMeter()            # 손실 값 집계
    sent_count = AverageMeter()     # 문장 처리 개수 집계
    
    # 학습 모드로 교체
    model.train()
    scaler = GradScaler()
    start = end = time.time()
    
    for step, batch in enumerate(train_loader):        
        data_time.update(time.time() - end)
        
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(args.device)
        
        batch_size = list(batch.values())[0].size(0)    
        
        try:
            with autocast(enabled=args.fp16):                
                res  = model(batch)
                loss = res['loss'].mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()    
            optimizer.zero_grad() 
            if args.scheduler in ['constant', 'onecycle', 'linear', 'cosine']:
                scheduler.step()                
            # loss 값을 기록
            losses.update(loss.item(), batch_size)
            maes.update(res['mae'].mean().item(), batch_size)
        except Exception as e:
            print(e)
            grad_norm = 0


        # 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        # CFG.print_freq 주기대로 결과 로그를 출력
        if (args.rank == 0) and  (step % args.print_freq == 0 or step == (len(train_loader)-1)):            
            print('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                  'MAE: {mae.val:.3f}({mae.avg:.3f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  'sent/s {sent_s:.0f} '
                  .format(
                   epoch, step+1, len(train_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses, mae=maes,
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   sent_s=sent_count.avg/batch_time.avg
                   ))
    res = {'loss': losses.avg, 'mae':maes.avg}
    return res


def valid(valid_loader, model, args):
        # AverageMeter는 지금까지 입력 받은 전체 수의 평균 값 반환 용도
    batch_time = AverageMeter()     # 한 배치처리 시간 집계
    data_time = AverageMeter()      # 데이터 로딩 시간 집계
    losses = AverageMeter()            # 손실 값 집계        
    maes = AverageMeter()            # 손실 값 집계
    sent_count = AverageMeter()     # 문장 처리 개수 집계
        
    model.eval()    
    start = end = time.time()
    
    prediction = []
    breath_id = []
    
    for step, batch in enumerate(valid_loader):
        # 데이터 로딩 시간 기록
        data_time.update(time.time() - end)
        
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(args.device)
        
        batch_size = list(batch.values())[0].size(0)    
        
        with torch.no_grad():
            with autocast(enabled=args.fp16):                
                res  = model(batch)
                #loss = res['loss'].mean()
                loss = res['loss'].mean()
                prediction.append(res['prediction'].cpu())
                breath_id.append(batch['breath_id'].cpu())                
        losses.update(loss.item(), batch_size)
        maes.update(res['mae'].mean().item(), batch_size)        
        
        # 소요시간 측정
        batch_time.update(time.time() - end)
        end = time.time()

        sent_count.update(batch_size)

        # CFG.print_freq 주기대로 결과 로그를 출력
        if (args.rank == 0) and  (step % args.print_freq == 0 or step == (len(valid_loader)-1)):            
            print('Test: [{0}/{1}] '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.3f}({loss.avg:.3f}) '
                  'MAE: {mae.val:.3f}({mae.avg:.3f}) '
                  'sent/s {sent_s:.0f} '
                  .format(
                   step+1, len(valid_loader), batch_time=batch_time,                   
                   data_time=data_time, loss=losses, mae=maes,
                   remain=timeSince(start, float(step+1)/len(valid_loader)),
                   sent_s=sent_count.avg/batch_time.avg
                   ))
        
    prediction = torch.cat(prediction).numpy()
    breath_id = torch.cat(breath_id).numpy()    
    res = {'loss': losses.avg, 'mae':maes.avg, 'prediction':prediction, 'breath_id':breath_id}
    return res


def save_checkpoint(state, model_path):
    print('saving cust_model ...')
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, model_path)  
    


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))
    

if __name__ == '__main__':
    main()
