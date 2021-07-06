#!/bin/bash
#SBATCH --job-name=300e2e_t2v
#SBATCH --output=log/t2vLRW.out.%j
#SBATCH --error=log/t2vLRW.out.%j
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:1
srun --mem=128GB  --ntasks=1 --gres=gpu:p6000:1 bash -c " 
          source /cfarhomes/peratham/.bashrc.tf3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/video_OSL_300;\
          time  python main.py --path './backendGRU_every_frame/backendGRU_4.pt' --dataset '/vulcan/scratch/peratham/lrw/e2e_npz' \
                                       --mode 'backendGRU' --every-frame \
                                       --batch-size=36 --lr=3e-4 \
                                       --epochs=5 --resume \
				       --resume-epochs=5 --workers=0;
" &
wait

