import pandas as pd
import matplotlib.pyplot as plt
import re

# Load data from a file (assume file is named 'strategy_data.csv')
# File should have two columns: 'Training Set Size' and 'Accuracy'
file_paths = [#'results/uc_merced_BALDDropout_100_100_1000_normal_res.txt',
               'results/uc_merced_KMeansSampling_100_100_1000_normal_res.txt',
              'results/uc_merced_RandomSampling_100_100_1000_normal_res.txt'#,
            #   'results/uc_merced_BadgeSampling_100_100_1000_normal_res.txt'
              ]

#read last 11 lines of the file
def read_last_11_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        last_11_lines = lines[-11:]
    return last_11_lines

all_values = {}

plotting = 'comparison' # 'comparison' or 'single'

for file_path in file_paths:
    lines = read_last_11_lines(file_path)
    data = {'Training Set Size': [], 'Accuracy': []}

    for line in lines:
        print(line, end='')
        values = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        data['Training Set Size'].append(int(values[0]))
        data['Accuracy'].append(float(values[1])*100)
    # Step 3: Convert to a DataFrame
    df = pd.DataFrame(data)

    # Load data
    data = df
    # add data to a dictionary
    all_values[file_path.split('_')[2]] = {'Training Set Size': data['Training Set Size'], 'Accuracy': data['Accuracy']}

    if file_path.split('_')[2] == 'BALDDropout' :
        color = '#2ca02c' 
    elif file_path.split('_')[2] == 'KCenterGreedy' :
        color = '#9467bd' 
    elif file_path.split('_')[2] == 'BadgeSampling' :
        color = '#ff7f0e'
    elif file_path.split('_')[2] == 'KMeansSampling' :
        color = '#d62728'   
    else :
        None
    if plotting == 'single':
        # Plot graph
        plt.figure(figsize=(10, 6))
        plt.plot(data['Training Set Size'], data['Accuracy'], marker='o', label=file_path.split('_')[2],color=color)

        # Add annotations
        plt.axvline(x=1000, color='red', linestyle='--', label='Quota Limit (1000)')
        plt.title('Accuracy vs. Training Set Size', fontsize=16)
        plt.xlabel('Training Set Size', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        #show numerical values on the plot points
        for i, txt in enumerate(data['Accuracy']):
            plt.annotate(f'{txt:.2f}', (data['Training Set Size'][i], data['Accuracy'][i]), fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save and display the plot
        output_path = 'results_plot/annotated/Acc_vs_TrainSet_size_'+file_path.split('_')[2]+'.png'
        #plt.savefig(output_path)
        plt.show()

        print(f"Graph saved to {output_path}")
if plotting == 'comparison':
    # Plot graph
    plt.figure(figsize=(10, 6))
    for strategy, values in all_values.items():
        if strategy == 'RandomSampling':
            color = '#1f77b4'
        else:
            if strategy == 'BALDDropout' :
                color = '#2ca02c' 
            elif strategy == 'KCenterGreedy' :
                color = '#9467bd' 
            elif strategy == 'BadgeSampling' :
                color = '#ff7f0e'
            elif strategy == 'KMeansSampling' :
                color = '#d62728'   
            else :
                None
        plt.plot(values['Training Set Size'], values['Accuracy'], marker='o', label=strategy,color=color)

    # Add annotations
    plt.axvline(x=1000, color='red', linestyle='--', label='Quota Limit (1000)')
    plt.title('Accuracy vs. Training Set Size', fontsize=16)
    plt.xlabel('Training Set Size', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # draw horizontal lines at every 5% accuracy
    for i in range(20, 95, 5):
        plt.axhline(y=i, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    
    #show numerical values on the plot points that are larger than the ones of RandomSampling
    # for strategy, values in all_values.items():
    #     if strategy != 'RandomSampling':
    #         for i, txt in enumerate(values['Accuracy']):
    #             if txt > all_values['RandomSampling']['Accuracy'][i]:
    #                 plt.annotate(f'{txt:.2f}', (values['Training Set Size'][i], values['Accuracy'][i]), fontsize=12)

    # Save and display the plot
    output_path = 'results_plot/annotated/Acc_vs_TrainSet_size_comparison.png'
    #plt.savefig(output_path)
    plt.show()

    print(f"Graph saved to {output_path}")  
