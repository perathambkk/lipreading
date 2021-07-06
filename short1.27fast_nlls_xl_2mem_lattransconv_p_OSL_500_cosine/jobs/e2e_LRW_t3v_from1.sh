#!/bin/bash
#SBATCH --job-name=pt300e2e_t3v
#SBATCH --output=log/t3vLRW.out.%j
#SBATCH --error=log/t3vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --gres=gpu:p6000:2
srun --mem=64GB  --ntasks=1 --gres=gpu:p6000:2 bash -c " 
          source /cfarhomes/peratham/.bashrc.th3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/transformer_OSL_300_parallel;\
          time  python main.py --path '/vulcan/scratch/peratham/swpath/end-to-end-lipreading/video_OSL_300/temporalConv/temporalConv_27.pt' \
          								--dataset '/vulcan/scratch/peratham/lrw/e2e_npz' \
                                       --mode 'finetuneSelfAttention' --every-frame \
                                       --batch-size=36 --lr=3e-5 \
                                       --epochs=30;
           " &
wait
