export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
deepspeed --num_gpus=4 pretrain.py --train_args_file hparams/train_args.json