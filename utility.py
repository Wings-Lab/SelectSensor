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


if __name__ == '__main__':
    dic = read_config('config.json')
    print(dic)
