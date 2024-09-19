#!/bin/bash
#SBATCH --job-name=bovesp_pre_pandemic
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o job.log
#SBATCH --output=job_pre_pandemic_output.txt
#SBATCH --error=job_pre_pandemic_error.txt

# carregar versão python
module load Python/3.8

# ativar ambiente
source env/bin/activate

# executar .py
python3 src/ibovespa/test.py --period pre-pandemic
