#!/bin/bash
#SBATCH --job-name=clt1v_nl_cos_500
#SBATCH --output=log/t1vLRW.out.%j
#SBATCH --error=log/t1vLRW.out.%j
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --partition=dpart
#SBATCH --qos=high
#SBATCH --gres=gpu:4
#SBATCH --exclude=vulcan19
LR=0.000225
echo "lr: ${LR}"
srun --mem=64GB  --ntasks=1 --gres=gpu:4 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_nl_lattransconv_p_OSL_500_cosine;\
          time  python maint1.py --path '' \
          	--dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       --mode 'temporalConv' \
                                       --batch-size=96 --lr=${LR} \
                                       --epochs=30 --workers=16;

" &
wait

