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


def figure_1a(selectsensor):
    '''Y - Probability of error
       X - # of sensor
       Offline + Homogeneous
       Algorithm - greedy, coverage, and random
    '''
    #plot_data = selectsensor.select_offline_coverage(70, 4)
    #save_data(plot_data, 'plot_data64/Offline_Coverage_2.csv')

    #plot_data = selectsensor.select_offline_random(100, 4)
    #save_data(plot_data, 'plot_data64/Offline_Random.csv')

    plot_data = selectsensor.select_offline_greedy_p_lazy(40, 4)
    save_data_offline_greedy(plot_data, 'plot_data64/Offline_Greedy.csv')


def figure_1b(selectsensor):
    '''Y - Probability of error
       X - Total budget
       Offline + Heterogeneous
       Algorithm - greedy, coverage, and random
    '''
    #plot_data = selectsensor.select_offline_random_hetero(40, 6)
    #save_data(plot_data, 'plot_data64/Offline_Random_hetero.csv')

    #plot_data = selectsensor.select_offline_coverage_hetero(30, 6)
    #save_data(plot_data, 'plot_data64/Offline_Coverage_hetero.csv')

    plot_data = selectsensor.select_offline_greedy_hetero_lazy(20, 12)
    save_data(plot_data, 'plot_data64/Offline_Greedy_hetero.csv')


def figure_2a(selectsensor):
    '''Y - empirical accuracy
       X - # of sensors selected
       Online + Homogeneous
       Algorithm - greedy + nearest + random
    '''
    plot_data = selectsensor.select_online_nearest(10, 48, 0)
    save_data(plot_data, 'plot_data20/Online_Nearest.csv')

    plot_data = selectsensor.select_online_random(20, 48, 0)
    save_data(plot_data, 'plot_data20/Online_Random.csv')

    plot_data = selectsensor.select_online_greedy_p(6, 48, 0)
    save_data(plot_data, 'plot_data20/Online_Greedy.csv')


def figure_2b(selectsensor):
    '''Y - empirical accuracy
       X - # of sensors selected
       Online + Heterogeneous
       Algorithm - greedy + nearest + random
    '''
    plot_data = selectsensor.select_online_random_hetero(25, 48, 769)
    save_data(plot_data, 'plot_data30/Online_Random_30_hetero.csv')

    plot_data = selectsensor.select_online_nearest_hetero(20, 48, 769)
    save_data(plot_data, 'plot_data30/Online_Nearest_30_hetero.csv')

    plot_data = selectsensor.select_online_greedy_hetero(8, 48, 769)
    save_data(plot_data, 'plot_data30/Online_Greedy_30_hetero.csv')


if __name__ == '__main__':
    pass
