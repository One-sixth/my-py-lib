import json
from .path_tool import open2


def load_json(p):
    '''从文件加载JSON对象'''
    return json.load(open2(p, 'r'))


def save_json(obj, p):
    '''保存JSON对象到文件，保留中文字符'''
    json.dump(obj, open2(p, 'w'), ensure_ascii=False, indent=2)
