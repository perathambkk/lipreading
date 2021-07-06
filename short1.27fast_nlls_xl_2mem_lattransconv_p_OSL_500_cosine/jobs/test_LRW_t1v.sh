#!/bin/bash
#SBATCH --job-name=test1v_cos_500
#SBATCH --output=tlog/t1vLRW.out.%j
#SBATCH --error=tlog/t1vLRW.out.%j
#SBATCH --mem=48GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --gres=gpu:1
#SBATCH --exclude=vulcan19
P=10
echo "Epoch: ${P}"
srun --mem=48GB  --ntasks=1 --gres=gpu:1 bash -c " 
          source /cfarhomes/peratham/.bashrc.th1.3.1;\
      cd /vulcan/scratch/peratham/swpath/end-to-end-lipreading/short15fast_conv2_p_OSL_500_cosine;\
          time  python main.py --path './temporalConv/temporalConv_${P}.pt' \
			--dataset '/vulcan/scratch/peratham/lrw-repro/shf_npz_rgb' \
                                       --mode 'temporalConv' --workers=12 \
                                       --batch-size=36 --lr=1.75e-4 \
                                       --epochs=40 --test;
           " &
wait
