export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --include=localhost:2,3 pretrain.py --train_args_file hparams/train_args.json