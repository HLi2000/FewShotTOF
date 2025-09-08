source activate rec_ood
module load CUDA/12.4.1 fsl

pwd
nvidia-smi
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
python ./src/eval.py experiment=ixi_eval.yaml