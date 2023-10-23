import json
from .path_tool import open2


def load_json(p):
    return json.load(open2(p, 'r'))


def save_json(obj, p):
    json.dump(obj, open2(p, 'w'), ensure_ascii=False, indent=2)
