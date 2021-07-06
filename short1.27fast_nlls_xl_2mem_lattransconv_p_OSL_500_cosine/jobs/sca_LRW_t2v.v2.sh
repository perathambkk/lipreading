#!/bin/bash
#SBATCH --job-name=clt2v_cos_500
#SBATCH --output=log/t2vLRW.out.%j
#SBATCH --error=log/t2vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
#SBATCH --exclude=vulcan00,vulcan19
LR=0.00025
echo "lr: ${LR}"
srun --mem=64GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_lattransconv_p_OSL_500_cosine;\
          time  python main.py --path '/vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_conv2_p_OSL_500_cosine/temporalConv/temporalConv_15.pt' \
          							--dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       --mode 'backendSelfAttention' \
                                       --batch-size=84 --lr=${LR} \
                                       --epochs=30 --workers=16;

" &
wait

