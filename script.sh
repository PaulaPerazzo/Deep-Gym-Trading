#!/bin/bash
#SBATCH --job-name=bovesp_all_test
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_all_output.txt
#SBATCH --error=job_all_error.txt

# carregar versão python
module load Python/3.8

# ativar ambiente
source env/bin/activate

# executar .py
python3 src/ibovespa/test.py --period all
