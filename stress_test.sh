cd code
python -m torch.distributed.launch --nproc_per_node=4 --master_port=40010 train.py --fold 0 --seed 7 --batch_size 1024 --num_workers 4 --arch LSTM --lr 1e-03 --max_epochs 1000 --fp16 --max_seq_len 80 --dropout 0.1
