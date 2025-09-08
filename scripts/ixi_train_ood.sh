source activate rec_ood
module load CUDA/12.4.1 fsl

pwd
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
wandb login
python ./src/train.py logger=many_loggers experiment=ixi_train.yaml