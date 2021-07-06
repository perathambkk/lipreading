#!/bin/bash
#SBATCH --job-name=clt3v_nl_cos_500
#SBATCH --output=log/t3vLRW.out.%j
#SBATCH --error=log/t3vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:4
#SBATCH --nodelist=vulcan07
#SBATCH --exclude=vulcan19
LR=0.000225
echo "lr: ${LR}"
srun --mem=64GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_nl_lattransconv_p_OSL_500_cosine_v2;\
          time  python maint3.py --path '/vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_nl_p_OSL_500_cosine/backendSelfAttention_every_frame/backendSelfAttention_5.pt' \
          							--dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       --mode 'finetuneSelfAttention' --every-frame \
                                       --batch-size=92 --lr=${LR} \
                                       --epochs=40 --workers=16;

" &
wait

