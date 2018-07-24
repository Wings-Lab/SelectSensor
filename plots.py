'''
Plots
'''

def save_data(plot_data, file_path):
    '''Save the plot_data to file_path
    Attributes:
        plot_data (list)
        file_path (str)
    '''
    with open(file_path, 'w') as f:
        for data in plot_data:
            print(data)
            f.write(str(data[1]) + ',' + str(data[2]) + '\n')


def figure_1a():
    '''Read data from file_path and plot it.
       X - # of sensors
       Y - Probability of error (1 - O_T)
    Attributes:
        plot_data (str): file that stores the data
    '''


if __name__ == '__main__':
    figure_1a()
