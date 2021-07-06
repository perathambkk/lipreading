#!/bin/bash
#SBATCH --job-name=t3v_cos_500
#SBATCH --output=log/t3vLRW.out.%j
#SBATCH --error=log/t3vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:2
#SBATCH --exclude=vulcan19
srun --mem=64GB  --ntasks=1 --gres=gpu:2 bash -c " 
          source /cfarhomes/peratham/.bashrc.th3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/p_transformer_OSL_500_cosine;\
          time  python main.py --path './backendSelfAttention_every_frame/backendSelfAttention_5.pt' \
			--dataset '/vulcan/scratch/peratham/lrw-repro/npz' \
                                       --mode 'finetuneSelfAttention' --every-frame \
                                       --batch-size=72 --lr=1.75e-4 \
                                       --epochs=40;
           " &
wait
