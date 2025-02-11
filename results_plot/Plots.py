# We have many strategies . (21 strategies)
# we want to showcase the results of the 10 rounds for each strategy in other color
import pandas as pd
from collections import deque
import re
import os
import matplotlib.pyplot as plt
import numpy as np

def read_last_2_lines(file_path):
    with open(file_path, 'r') as file:
        # Use deque to keep only the last 5 lines
        last_2_lines = deque(file, maxlen=2)
    return list(last_2_lines)

def read_7th_and_6th_lines_from_end(file_path):
    with open(file_path, 'r') as file:
         # Use deque to keep only the last 7 lines
        last_7_lines = deque(file, maxlen=7)
    # Return the 7th and 6th lines from the end
    return list(last_7_lines)[-7], list(last_7_lines)[-6]


# List of input files
input_files = [ 
                'uc_merced_RandomSampling_100_100_1000_normal_log.txt',
                #'uc_merced_BadgeSampling_100_100_1000_normal_log.txt',
                #'uc_merced_BALDDropout_100_100_1000_normal_log.txt',
                'uc_merced_KMeansSampling_100_100_1000_normal_log.txt'#,
                #'uc_merced_KCenterGreedy_100_100_1000_normal_log.txt'
                
                ]
#   - adversarial_bim                                   [ ]
#   - adversarial_deepfool                              [ ]
#   - badge_sampling                                    [✔️]
#   - bayesian_active_learning_disagreement_dropout     [✔️]
#   - ceal                                              [ ]
#   - entropy_sampling                                  [ ]
#   - entropy_sampling_dropout                          [ ]
#   - kcenter_greedy                                    [✔️]
#   - kcenter_greedy_pca                                [ ]
#   - kmeans_sampling                                   [✔️]
#   - kmeans_sampling_gpu                               [ ]
#   - least_confidence                                  [ ]
#   - least_confidence_dropout                          [ ]
#   - loss_prediction                                   [ ]
#   - margin_sampling                                   [ ]
#   - margin_sampling_dropout                           [ ]
#   - mean_std                                          [ ]
#   - random_sampling                                   [✔️]
#   - strategy                                          [ ]
#   - vaal                                              [ ]
#   - var_ratio                                         [ ]
#   - waal                                              [ ]

# Initialize a dictionary to store values from each file
all_values = {}

for input_str in input_files:
    strategy = input_str.split('_')[2]
    file_path = 'logfile/' + input_str
    if strategy != 'KCenterGreedy':
        last_2_lines = read_7th_and_6th_lines_from_end(file_path)
    else:
        last_2_lines = read_last_2_lines(file_path)
    lines = []
    for line in last_2_lines:
        print(line, end='')
        lines.append(line)

    print("RESULTS:\n", lines)
    input_string = lines[0] + lines[1]
    values = re.findall(r"[-+]?\d*\.\d+|\d+", input_string)
    float_values = [float(value) * 100 for value in values]
    
    # Store the values in the dictionary
    all_values[input_str] = float_values


# Plot the values from all files
plt.figure(figsize=(10, 6))
#df for mean and std with columns file name and mean and std
mean_std = []
color = '#1f77b4'
for file_name, values in all_values.items():
    values = np.array(values)
    x = np.arange(len(values))
    if file_name.split('_')[2] == 'BALDDropout' :
        color = '#2ca02c' 
    elif file_name.split('_')[2] == 'KCenterGreedy' :
        color = '#9467bd' 
    elif file_name.split('_')[2] == 'BadgeSampling' :
        color = '#ff7f0e'
    elif file_name.split('_')[2] == 'KMeansSampling' :
        color = '#d62728'   
    else :
        None
    #plt.plot(x, values, "-o", label=strategy, color=color)
    plt.plot(x, values, "-o", label=file_name.split('_')[2],color=color)
    # calc mean - std
    mean = np.mean(values)
    std = np.std(values)
    # append to df
    mean_std.append([file_name.split('_')[2], mean, std])
print("________________________________________________________")
print(mean_std)
#save to csv
df = pd.DataFrame(mean_std, columns=['strategy', 'mean', 'std'])
df.to_csv('results_plot/mean_std_10ep.csv', index=False)



plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#show numerical values on the plot points
for file_name, values in all_values.items():
    for i, txt in enumerate(values):
        plt.annotate('{:.2f}'.format(txt), (i, values[i] + 1.2))
plt.title('Accuracy Scores (%)\n of Strategies on UCMerced', fontweight='bold')
plt.show()















# input_str = 'uc_merced_KMeansSampling_100_100_1000_normal_log.txt'

# for file in os.listdir('logfile'):
#     if file.startswith('uc_merced'):
#         print(file)
#         input_str = file
#         strategy = input_str.split('_')[2]
#         file_path = 'logfile/'+ input_str
#         last_2_lines = read_last_2_lines(file_path)
#         for line in last_2_lines:
#             print(line, end='')
#             lines.append(line)

#         print("RESULTS:\n", lines)
#         input_string = lines[0]+lines[1] 
#         values = re.findall(r"[-+]?\d*\.\d+|\d+", input_string)
#         # Convert the string values to floats
#         float_values = [float(value)*100 for value in values]

#         # PLOT THE VALUES 
#         float_values = np.array(float_values)
#         x = np.arange(0, 11, 1)
#         plt.plot(x, float_values,"-o")   
#         plt.legend([strategy])

#         plt.plot(x, float_values, "s")
#         plt.xticks(x)
#         plt.xlabel('Round')
#         plt.ylabel('Accuracy')
#         plt.ylim(0, 100)
#         plt.title('Dataset: UC Merced ',fontweight='bold')  

#         for i, txt in enumerate(float_values):
#             plt.annotate('{:.2f}'.format(txt), (x[i]+0.2, float_values[i]+1.2))
#         plt.plot(float_values,)
#         plt.savefig('results_plot/'+input_str+'.png')
#         plt.show()
