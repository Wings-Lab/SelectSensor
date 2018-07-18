'''
Some useful utilities
'''
import json

def read_config(filename):
    '''Read json file into json object
    '''
    with open(filename, 'r') as handle:
        dictdump = json.loads(handle.read(), encoding='utf-8')
    return dictdump


def ordered_insert(sensor_index, index):
    '''Insert index into sensor_index and guarantee sensor_index is sorted from small to large.
    Attributes:
        sensor_index (list)
        index (int)
    '''
    size = len(sensor_index)
    for i in range(size):
        if index < sensor_index[i]:
            sensor_index.insert(i, index)
            break
    else:
        sensor_index.insert(size, index)


if __name__ == '__main__':
    dic = read_config('config.json')
    print(dic)
    sensor_list = [2, 4, 6, 8]
    ordered_insert(sensor_list, 11)
    print(sensor_list)
    sensor_list.remove(11)
    print(sensor_list)
