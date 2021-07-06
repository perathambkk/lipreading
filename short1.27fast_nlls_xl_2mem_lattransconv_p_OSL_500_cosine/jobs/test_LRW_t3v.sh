#!/bin/bash
#SBATCH --job-name=test3v_cos_500
#SBATCH --output=tlog/t3vLRW.out.%j
#SBATCH --error=tlog/t3vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:1
#SBATCH --exclude=vulcan19
P=10
echo "Epoch: ${P}"
srun --mem=64GB  --ntasks=1 --gres=gpu:1 bash -c " 
          source /cfarhomes/peratham/.bashrc.th3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/p_transformer_OSL_500_cosine;\
          time  python main.py --path './finetuneSelfAttention_every_frame/finetuneSelfAttention_${P}.pt' \
			--dataset '/vulcan/scratch/peratham/lrw-repro/npz' \
                                       --mode 'finetuneSelfAttention' --every-frame \
                                       --batch-size=36 --lr=1.75e-4 \
                                       --epochs=40 --test;
           " &
wait
