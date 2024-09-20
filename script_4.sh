#!/bin/bash
#SBATCH --job-name=dj_all
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -o job.log
#SBATCH --output=job_all_output.txt
#SBATCH --error=job_all_error.txt

# carregar versão python
module load Python/3.8

# ativar ambiente
source env/bin/activate

# executar .py
python3 src/dow_jones/test.py --period all
