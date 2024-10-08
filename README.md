# Deep Gym Trading

This repository contains the code developed for the research **"AI-Driven Index Tracking: Reinforcement Learning Applied to the Brazilian Market"**, which aims to adapt the Deep Trader framework to the Brazilian financial market using Reinforcement Learning (RL) for dynamic portfolio management. The code was developed using the OpenAI Gym library and Actor-Critic networks, focusing on tracking the Ibovespa index from 2000 to 2023.

## Table of Contents
 - [Installation](#instalation)
 - [Usage](#usage)
 - [Acknowledgment](#acknowledgment)
 - [License](#license)

## Instalation
### Requirements
 - Python 3.9
 - Libraries: in `requirements.txt` file

### Steps
1. Clone the repository:
```bash
git clone https://github.com/PaulaPerazzo/Deep-Gym-Trading.git
```

2. (Optional) Create and activate a virtual environment:
```python
python3 -m venv venv
source venv/bin/activate
```

3. Install the dependencies:
```python
pip install -r requirements.txt
```

4. Download historical financial data. Inside `./data` folder is an example.

## Usage
1. (Ibovespa) Run the main script to start training the RL agent:
```python
python3 src/ibovespa/main.py
```

Obs.: Note that there is a file to each epoch.

2. The model will be trained using historical Ibovespa data, and you can monitor the progress in the console. 

3. (Testing) Run the script named `test.py` to test the trained agent:
```python
python3 src/ibovespa/test.py --period <period>
```

## Acknowledgment
I would like to express my gratitude to the following people and organizations for their contributions to this project: 
 - Adiel T. de Almeida Filho, for his invaluable guidance and support throughout the research process.
 - Centro de Informática - UFPE, for providing computational resources and an inspiring research environment, along with the Cluster Apuana.
 - All contributors to the financial data provided via the Yahoo Finance API.

This research would not have been possible without the support and resources of these individuals and organizations.

## License
This project is licensed under the MIT License.
