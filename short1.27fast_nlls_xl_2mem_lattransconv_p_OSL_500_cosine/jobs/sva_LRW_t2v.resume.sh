#!/bin/bash
#SBATCH --job-name=clt2v_nl_cos_500
#SBATCH --output=log/t2vLRW.out.%j
#SBATCH --error=log/t2vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
#SBATCH --exclude=vulcan19
LR=0.000225
echo "lr: ${LR}"
srun --mem=64GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_nlls_2mem_lattransconv_p_OSL_500_cosine;\
          time  python main.py --path './backendSelfAttention_last_frame/backendSelfAttention_8.pt' \
          							--dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       --mode 'backendSelfAttention' --resume \
                                       --batch-size=92 --lr=${LR} --resume-epochs=9 \
                                       --epochs=10 --workers=16;

" &
wait

