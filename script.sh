#!/bin/bash
#SBATCH --job-name=b3_test
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH --gres=gpu:2
#SBATCH -c 4
#SBATCH -o job.log
#SBATCH --output=job_output_all.txt
#SBATCH --error=job_error_all.txt

# carregar versão python
module load Python/3.9

# ativar ambiente
source ./venv/bin/activate

# executar .py
python3 src/ibovespa/main.py --period all 
