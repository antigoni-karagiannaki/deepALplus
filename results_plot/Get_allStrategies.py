# get all strategies from the query_strategies folder
# then write the names of the strategies in a file

import os
import re

for file in os.listdir('query_strategies'):
    if file.endswith('.py'):
        strategy_name = file.split('.')[0]
        with open('results_plot/strategies.txt', 'a') as f:
            f.write(strategy_name + '\n')

print('Done!')
