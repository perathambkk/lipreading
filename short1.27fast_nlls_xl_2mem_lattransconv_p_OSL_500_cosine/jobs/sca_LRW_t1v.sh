#!/bin/bash
#SBATCH --job-name=sh15f_cb2_t1v
#SBATCH --output=log/t1vLRW.out.%j
#SBATCH --error=log/t1vLRW.out.%j
#SBATCH --mem=48GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
#SBATCH --exclude=vulcan03,vulcan19
LR=0.00025
echo "lr: ${LR}"
srun --mem=48GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_conv2_p_OSL_500_cosine;\
          time  python main.py --path '' --dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       	--mode 'temporalConv' \
					--workers=16 \
				--batch-size=84 --lr=${LR} \
                                       --epochs=40;
           " &
wait
