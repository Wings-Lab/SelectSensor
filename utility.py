import json

def read_config(filename):
    with open(filename, 'r') as handle:
        dictdump = json.loads(handle.read(), encoding='utf-8')
    return dictdump


if __name__ == '__main__':
    dic = read_config('config.json')
    print(dic)