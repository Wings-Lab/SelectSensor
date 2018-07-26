'''
Plots
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def save_data(plot_data, file_path):
    '''Save the plot_data to file_path
    Attributes:
        plot_data (list)
        file_path (str)
    '''
    print('start saving data')
    with open(file_path, 'w') as f:
        for data in plot_data:
            #print(data)
            f.write(str(data[1]) + ',' + str(data[2]) + '\n')


def figure_1a(greedy_file, coverage_file, random_file):
    '''Read data from file_path and plot it.
       X - # of sensors
       Y - Probability of error (1 - O_T)
    Attributes:
        greedy_file (str): file that stores the greedy algo result
        coverage_file (str):
        random_file (str)
    '''
    df_greedy = pd.read_csv(greedy_file, header=None)
    df_coverage = pd.read_csv(coverage_file, header=None)
    df_random = pd.read_csv(random_file, header=None)

    X_greedy, Y_greedy = df_greedy[0].tolist(), df_greedy[1].tolist()
    X_coverage, Y_coverage = df_coverage[0].tolist(), df_coverage[1].tolist()
    X_random, Y_random = df_random[0].tolist(), df_random[1].tolist()

    plt.figure(figsize=(14, 9))
    plt.plot(X_greedy, Y_greedy)
    plt.plot(X_coverage, Y_coverage)
    plt.plot(X_random, Y_random)
    plt.legend(['Greedy Selection', 'Coverage-based Selection', 'Random Selection'], prop={'size':16})
    plt.xlabel('Number of Sensors Selected', fontsize=16)
    plt.ylabel('O_T = 1-Prob(error)', fontsize=16)
    plt.title('Comparation of Three Selection Algorithms (15*15 Grid, 20 Sensors)', fontsize=20)
    x_axis = np.arange(1, 21, 1)
    plt.xticks(x_axis)
    y_axis = np.arange(0, 1.1, 0.1)
    plt.yticks(y_axis)
    plt.savefig('plot/offline_homo_15.png')
    plt.show()


if __name__ == '__main__':
    figure_1a('plot_data2/Offline_Greedy_15.csv', \
              'plot_data2/Offline_Coverage_15.csv', \
              'plot_data2/Offline_Random_15.csv')
