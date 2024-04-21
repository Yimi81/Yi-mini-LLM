export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --include=localhost:2,3,4,5,6,7 pretrain.py --train_args_file hparams/train_args.json