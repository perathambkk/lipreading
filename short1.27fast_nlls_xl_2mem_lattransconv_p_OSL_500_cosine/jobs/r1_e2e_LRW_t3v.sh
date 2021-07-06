#!/bin/bash
#SBATCH --job-name=300e2e_t3v
#SBATCH --output=log/t3vLRW.out.%j
#SBATCH --error=log/t3vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --partition=dpart
#SBATCH --qos=medium
#SBATCH --gres=gpu:p6000:1
srun --mem=64GB  --ntasks=1 --gres=gpu:p6000:1 bash -c " 
          source /cfarhomes/peratham/.bashrc.tf3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/video_OSL_300;\
          time  python main.py --path './finetuneGRU_every_frame/finetuneGRU_18.pt' --dataset '/vulcan/scratch/peratham/lrw/e2e_npz' \
                                       --mode 'finetuneGRU' --every-frame \
                                       --batch-size=36 --lr=3e-4 \
                                       --epochs=30 --resume \
				       --resume-epochs=19 --workers=0;
" &
wait

