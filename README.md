# poker-bot

Final project for CS 364M. The environment for our Texas Hold 'Em DQN was cloned from [neuron poker](https://github.com/dickreuter/neuron_poker).

## Envrionment Setup
- Install Python 3.11, I would also recommend to install PyCharm.
- Install Poetry with ``curl -sSL https://install.python-poetry.org | python3 -`` and add it to your path
- Create a virtual environment with ``poetry env use python3.11``
- Activate it with ``poetry shell``
- Install all required packages with ``poetry install --no-root``

## Run the code
```
Usage:
  main.py selfplay random [options]
  main.py selfplay keypress [options]
  main.py selfplay consider_equity [options]
  main.py selfplay equity_improvement --improvement_rounds=<> [options]
  main.py selfplay dqn_train [options]
  main.py selfplay dqn_play [options]
  main.py selfplay ebaad_train [options]
  main.py selfplay ebaad_play [options]
  main.py learn_table_scraping [options]

options:
  -h --help                 Show this screen.
  -r --render               render screen
  -c --use_cpp_montecarlo   use cpp implementation of equity calculator. Requires cpp compiler but is 500x faster
  -f --funds_plot           Plot funds at end of episode
  --log                     log file
  --name=<>                 Name of the saved model
  --screenloglevel=<>       log level on screen
  --episodes=<>             number of episodes to play
  --stack=<>                starting stack for each player [default: 500].

```

## Navigation
- All code is run through [main](https://github.com/EbaadImran/poker-bot/blob/main/main.py)
- To naviage to all agents in the project, see [agents](https://github.com/EbaadImran/poker-bot/tree/main/agents)
- Our DDQN poker agent can be found in [agent_ebaad](https://github.com/EbaadImran/poker-bot/blob/main/agents/agent_ebaad.py)
- To view our equity based agent we trained against, see [agent_consider_equity](https://github.com/EbaadImran/poker-bot/blob/main/agents/agent_consider_equity.py)
- Our baseline for a randomized agent can be seen in [agent_random](https://github.com/EbaadImran/poker-bot/blob/main/agents/agent_random.py)
- The modified Texas Hold 'Em environment can be seen [gym_env](https://github.com/EbaadImran/poker-bot/tree/main/gym_env)
  - [env](https://github.com/EbaadImran/poker-bot/blob/main/gym_env/env.py) contains the environment itself
