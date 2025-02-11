##############################################################
#                      Plots_v2.py                           #
############################################################################################################################################################################                                                                                                                                                                     #
#   Plotting the data stored in /results.                                                                                                                                  #
#                                                                                                                                                                          #
#  -- <dataset>_<strategy>_100_100_1000_normal_res_tot.txt              | (1) |                                                                                            #
#  -- <dataset>_<strategy>_100_100_1000_normal_res.txt                  | (2) |                                                                                            #
#  -- <dataset>_<strategy>_100_100_1000_normal_log.txt                  | (3) |                                                                                            #                                                                                                                                                                         #
# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                                          
#  (1) e.g.:  .._res_tot                               #   (2) e.g.:   .._res                                   #  (3) e.g.:  .._normal_log                                                                                                                                                                                            
#       dataset: uc_merced                             #      dataset: uc_merced                                #       0: 0.655
#       AL strategy: BadgeSampling                     #      AL strategy: BadgeSampling                        #       1: 0.626        
#       number of labeled pool: 100                    #      number of labeled pool: 100                       #       2: 0.638  
#       number of unlabeled pool: 1160                 #      number of unlabeled pool: 1160                    #       mean AUBC(acc): 0.6397. std dev AUBC(acc): 0.0119       
#       number of testing pool: 420                    #      number of testing pool: 420                       #       mean time: 21090.0. std dev time: 373.3711             
#       batch size: 100                                #      batch size: 100                                   #                  
#       quota: 1000                                    #      quota: 1000                                       #                   
#       time of repeat experiments: 3                  #      time of repeat experiments: 3                     #              
#       0: 0.632                                       #      Size of training set is 100, accuracy is 0.2603.  #            
#       1: 0.59                                        #                          ...                           #
#       2: 0.636                                       #      Size of training set is 900, accuracy is 0.7286.  #
#       mean acc: 0.6193. std dev acc: 0.0208          #      Size of training set is 1000, accuracy is 0.781.  #
#       mean time: 20135.6667. std dev acc: 1713.4971  #      Size of training set is 1100, accuracy is 0.7873. #                                                                                                                            
#                                                      #                                                        #                    
############################################################################################################################################################################
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# Functions that reads the txt files from the '/results' file
def read_res_tot( dataset, strategy):
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_res_tot.txt'
    file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name),'r')
    lines = file_res_tot.readlines()
    file_res_tot.close()
    mean_acc = float(lines[-2].split(' ')[2])
    std_dev_acc = float(lines[-2].split(' ')[5])
    mean_time = float(lines[-1].split(' ')[2])
    std_dev_time = float(lines[-1].split(' ')[5])
    
    return mean_acc, std_dev_acc, mean_time, std_dev_time

def read_res( dataset, strategy):
    """Reads the results from the '_normal_res.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        acc (list): A list of accuracies. 
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_res.txt'
    file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name),'r')
    lines = file_res.readlines()
    file_res.close()
    acc = []
    for row in range(8,len(lines)): 
        number = lines[row].split(' ')[8] # 0.2603.\n
        # remove dot and new line
        number = number[:-2] 
        acc.append(float(number))

    return acc


# -------------------------------------------------------------------
# Function that plots the data
def plot_data( dataset, strategy, acc):
    """Plots the data from the '_normal_res.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.
        acc (list): A list of accuracies. 

    Returns:
        None
    """
    # Create a figure and a set of subplots
    fig, ax = plt.subplots(figsize=(12, 6))
    # Set the background color of the plot to white
    fig.patch.set_facecolor('white')
    # Set the title of the plot
    plt.title('Active Learning Strategy: ' + strategy + ' on ' + dataset, fontsize=16)
    # Set the x-axis label
    plt.xlabel('Number of Labeled Samples', fontsize=14)
    # Set the y-axis label
    plt.ylabel('Accuracy', fontsize=14)
    # Set the x-axis limits
    plt.xlim(0, 1100)
    # Set the y-axis limits
    plt.ylim(0, 1)
    # Set the x-axis ticks
    plt.xticks(np.arange(0, 1100, 100))
    # Set the y-axis ticks
    plt.yticks(np.arange(0, 1.1, 0.1))
    # Set the grid lines
    plt.grid(True)
    # Plot the data
    plt.plot(np.arange(0, 1100, 100), acc, color='blue', marker='o', linestyle='-', linewidth=2, markersize=6)
    # Show the plot
    plt.show()

# -------------------------------------------------------------------
def get_final_accuracy(dataset, strategy):
    """Reads the results from the '_normal_res.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        acc (value): The final accuracy.
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_res.txt'
    file_res =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name),'r')
    lines = file_res.readlines()
    file_res.close()
    acc = []
    row = 18
    number = lines[row].split(' ')[8] # 0.2603.\n
    # remove dot and new line
    number = number[:-2]
    acc.append(float(number))
    return acc 
def get_std_dev_acc(dataset, strategy):
    """Reads the results from the '_normal_res_tot.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        std_dev_acc (value): The standard deviation of the accuracy.
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_res_tot.txt'
    file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name),'r')
    lines = file_res_tot.readlines()
    file_res_tot.close()
    std_dev_acc = float(lines[-2].split(' ')[6])
    return std_dev_acc
def get_mean_time(dataset, strategy):
    """Reads the results from the '_normal_res_tot.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        mean_time (value): The mean time.
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_res_tot.txt'
    file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name),'r')
    lines = file_res_tot.readlines()
    file_res_tot.close()
    time = lines[-1].split(' ')[2]
    # remove dot 
    time = time[:-1]
    # round to 3 decimal places
    mean_time = float(time)/1000 # MILISECONDS to seconds
    mean_time = round(mean_time, 2)
    return mean_time
def get_mean_time_stddev(dataset, strategy):
    """Reads the results from the '_normal_res_tot.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        mean_time_stddev (value): The standard deviation of the time.
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_res_tot.txt'
    file_res_tot =  open(os.path.join(os.path.abspath('') + '/results', '%s' % file_name),'r')
    lines = file_res_tot.readlines()
    file_res_tot.close()
    time = lines[-1].split(' ')[6]
    # round to 3 decimal places
    mean_time_stddev = float(time)/1000 # MILISECONDS to seconds
    mean_time_stddev = round(mean_time_stddev, 2)
    return mean_time_stddev


# -------------------------------------------------------------------
# Functions that get from the /logfile folder the data about AUBC

def get_mean_aubc(dataset, strategy):
    """Reads the results from the '_normal_log.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        acc_m (list): A list of accuracies. 
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_log.txt'
    file_res_tot =  open(os.path.join(os.path.abspath('') + '/logfile', '%s' % file_name),'r')
    lines = file_res_tot.readlines()
    file_res_tot.close()
    aubc = lines[-2].split(' ')[2]
    # remove dot
    aubc = aubc[:-1]
    return aubc
def get_mean_aubc_stddev(dataset, strategy):    
    """Reads the results from the '_normal_log.txt' txt files in the '/results' folder.
     
    Args:
        dataset (str): The dataset name.
        strategy (str): The active learning strategy name.

    Returns:
        mean_aubc_stddev (value): The standard deviation of the AUBC.
    """
    file_name = dataset + '_'  + strategy + '_100_100_1000_normal_log.txt'
    file_res_tot =  open(os.path.join(os.path.abspath('') + '/logfile', '%s' % file_name),'r')
    lines = file_res_tot.readlines()
    file_res_tot.close()
    aubc = lines[-2].split(' ')[6]
    return aubc


# ------------------------------------------------------------------- 
# -------------------------- DEMO -----------------------------------

DATASET = 'uc_merced'
STRATEGY = 'BadgeSampling'
STRATEGY2 = 'EntropySampling'

f_acc = get_final_accuracy( DATASET ,  STRATEGY )
std_dev_acc = get_std_dev_acc( DATASET ,  STRATEGY )
mean_time = get_mean_time( DATASET ,  STRATEGY )
mean_time_stddev = get_mean_time_stddev( DATASET ,  STRATEGY )
aubc = get_mean_aubc( DATASET ,  STRATEGY2 )
std_aubc = get_mean_aubc_stddev( DATASET ,  STRATEGY2 )

print()
print(f'f-acc: \t \t  {f_acc}')
print(f'std_acc:\t  ± {std_dev_acc}')
print(f'mean_time:\t  {mean_time}')
print(f'mean_time_stddev: {mean_time_stddev}')
print(f'aubc: \t \t  {aubc}')
print(f'std_aubc: \t  ± {std_aubc}')