'''
Plots
'''

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


def save_data_offline_greedy(plot_data, file_path):
    '''Save the plot_data to file_path for offline greedy
       Both ot approx and ot real
    Attributes:
        plot_data (list)
        file_path (str)
    '''
    print('start saving data')
    with open(file_path, 'w') as f:
        for data in plot_data:
            # length, ot_approx, ot_real
            f.write(str(data[0]) + ',' + str(data[1]) + ',' + str(data[2]) + '\n')


if __name__ == '__main__':
    pass
