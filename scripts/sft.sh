export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --master_port 29502 --include=localhost:0,1 train.py --train_args_file hparams/sft_args.json