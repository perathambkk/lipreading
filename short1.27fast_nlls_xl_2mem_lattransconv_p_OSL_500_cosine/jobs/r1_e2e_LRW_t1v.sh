#!/bin/bash
#SBATCH --job-name=300e2e_t1v
#SBATCH --output=log/t1vLRW.out.%j
#SBATCH --error=log/t1vLRW.out.%j
#SBATCH --mem=32GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:p6000:1
srun --mem=32GB  --ntasks=1 --gres=gpu:p6000:1 bash -c " 
          source /cfarhomes/peratham/.bashrc.tf3.6;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/video_OSL_300;\
          time  python main.py --path './temporalConv/temporalConv_27.pt' --dataset '/vulcan/scratch/peratham/lrw/e2e_npz' \
                                       --mode 'temporalConv' \
                                       --batch-size=36 --lr=3e-4 \
                                       --epochs=30
					--resume --resume-epochs=28;
           " &
wait
