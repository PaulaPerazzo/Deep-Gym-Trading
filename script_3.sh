#!/bin/bash
#SBATCH --job-name=b3_test_pandemic
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH --gres=gpu:2
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_output_pandemic_2024-10-07.txt
#SBATCH --error=job_error_pandemic_2024-10-07.txt

# carregar versão python
module load Python/3.9

# ativar ambiente
source ./env/bin/activate

# executar .py
python3 src/ibovespa/test.py --period pandemic 
