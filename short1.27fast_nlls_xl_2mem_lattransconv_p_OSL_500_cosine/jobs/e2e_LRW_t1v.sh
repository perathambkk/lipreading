#!/bin/bash
#SBATCH --job-name=sf_cb_t1v
#SBATCH --output=log/t1vLRW.out.%j
#SBATCH --error=log/t1vLRW.out.%j
#SBATCH --mem=48GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:4
LR=0.000025
echo "lr: ${LR}"
srun --mem=48GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/slowfast_conv_p_OSL_500_cosine;\
          time  python main.py --path '' --dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       	--mode 'temporalConv' \
					--workers=16 \
				--batch-size=120 --lr=${LR} \
                                       --epochs=40;
           " &
wait
