export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --master_port 29503 --include=localhost:2,3,4,5,6,7 train.py --train_args_file hparams/sft_args.json