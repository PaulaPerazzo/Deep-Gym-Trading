#!/bin/bash
#SBATCH --job-name=dji_post_pandemic
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o job.log
#SBATCH --output=job_post_pandemic_output.txt
#SBATCH --error=job_post_pandemic_error.txt

# carregar versão python
module load Python/3.8

# ativar ambiente
source env/bin/activate

# executar .py
python3 src/dow_jones/main.py --period post-pandemic