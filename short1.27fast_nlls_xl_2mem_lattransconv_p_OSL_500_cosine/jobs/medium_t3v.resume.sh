#!/bin/bash
#SBATCH --job-name=clt3v_nl_cos_500
#SBATCH --output=log/t3vLRW.out.%j
#SBATCH --error=log/t3vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:p6000:2
#SBATCH --exclude=vulcan19
LR=0.0001125
echo "lr: ${LR}"
srun --mem=64GB  --ntasks=1 --gres=gpu:2 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short1.27fast_nlls_2mem_lattransconv_p_OSL_500_cosine;\
          time  python main.py --path './finetuneSelfAttention_every_frame/finetuneSelfAttention_3.pt' \
          							--dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       --mode 'finetuneSelfAttention' --every-frame --resume --resume-epochs=4 \
                                       --batch-size=32  --lr=${LR} \
                                       --epochs=30 --workers=16;

" &
wait

