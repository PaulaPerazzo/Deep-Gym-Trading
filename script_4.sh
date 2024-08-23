#!/bin/bash
#SBATCH --job-name=sp500_all
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_all_output.txt
#SBATCH --error=job_pre_all_error.txt

# carregar versão python
module load Python/3.8

# ativar ambiente
source env/bin/activate

# executar .py
python3 src/sp_500/main.py --period all