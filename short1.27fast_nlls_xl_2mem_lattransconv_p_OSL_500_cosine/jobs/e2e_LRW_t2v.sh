#!/bin/bash
#SBATCH --job-name=t2v_cos_500
#SBATCH --output=log/t2vLRW.out.%j
#SBATCH --error=log/t2vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
srun --mem=64GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/p_transformer_OSL_500_cosine;\
          time  python main.py --path '/vulcan/scratch/peratham/swpath/end-to-end-lipreading/video_112/temporalConv/temporalConv_20.pt' \
          							--dataset '/vulcan/scratch/peratham/lrw-repro/npz' \
                                       --mode 'backendSelfAttention' --every-frame \
                                       --batch-size=144 --lr=12e-4 \
                                       --epochs=5;
" &
wait

