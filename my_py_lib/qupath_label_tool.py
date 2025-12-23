import numpy as np
import geojson
from my_py_lib.json_tool import open2


def save_line_string_geojson(conts, colors, names, file, auto_close_ring=True):
    if colors is None:
        colors = [[255, 0, 0]] * len(conts)
    else:
        colors = [[255, 0, 0] if c is None else c for c in colors]

    if names is None:
        names = ['轮廓'] * len(conts)

    fs = []

    for cont, color, name in zip(conts, colors, names, strict=True):
        assert isinstance(cont, np.ndarray)
        # 自动关闭轮廓
        if auto_close_ring and np.any(cont[0] != cont[-1]):
            cont = np.concatenate([cont, cont[:1]], 0)
        cont = cont[:, ::-1].tolist()
        cont = geojson.LineString(coordinates=cont)
        f = geojson.Feature(geometry=cont, properties={
            'objectType': 'annotation',
            'color': color,
            'classification': {
                'name': name,
                'color': color,
            },
            'isLocked': False,
        })
        fs.append(f)

    fc = geojson.FeatureCollection(fs)
    geojson.dump(fc, open2(file, 'w'), indent=2, ensure_ascii=False)


def load_line_string_geojson(file):
    data = geojson.load(open2(file, 'r'))

    conts, colors, names = [], [], []

    if isinstance(data, list):
        features = data
    else:
        features = data.features

    for item in features:
        coords = list(geojson.utils.coords(item))
        coords = np.asarray(coords, np.float32)
        coords = coords[:, ::-1]
        try:
            cls_name = item.properties['classification']['name']
        except KeyError:
            cls_name = None
        color = None
        try:
            color = item.properties['classification']['color']
        except KeyError:
            pass
        try:
            color = item.properties['color']
        except KeyError:
            pass
        conts.append(coords)
        colors.append(color)
        names.append(cls_name)

    return conts, colors, names


def save_point_geojson(points, colors, names, file):
    if colors is None:
        colors = [[255, 0, 0]] * len(points)
    else:
        colors = [[255, 0, 0] if c is None else c for c in colors]

    if names is None:
        names = ['point'] * len(points)

    points = np.asarray(points).tolist()

    new_colors_data = {}
    for point, color, name in zip(points, colors, names, strict=True):
        color = tuple(color)
        ddd = new_colors_data.setdefault((color, name), [])
        ddd.append(point[::-1])

    fs = []

    for color_name, data in new_colors_data.items():
        color, name = color_name
        points = data
        mp = geojson.MultiPoint(coordinates=points)
        f = geojson.Feature(geometry=mp, properties={
            'objectType': 'annotation',
            'color': color,
            'classification': {
                'name': name,
                'color': color,
            },
            'isLocked': False,
        })
        fs.append(f)

    fc = geojson.FeatureCollection(fs)
    geojson.dump(fc, open2(file, 'w'), indent=2, ensure_ascii=False)


def load_point_geojson(file):
    data = geojson.load(open2(file, 'r'))

    points, colors, names = [], [], []

    if isinstance(data, list):
        features = data
    else:
        features = data.features

    for item in features:
        coords = list(geojson.utils.coords(item))
        coords = np.asarray(coords, np.float32)
        coords = coords[:, ::-1]
        try:
            cls_name = item.properties['classification']['name']
        except KeyError:
            cls_name = None
        color = None
        try:
            color = item.properties['classification']['color']
        except KeyError:
            pass
        try:
            color = item.properties['color']
        except KeyError:
            pass
        points.extend(coords)
        colors.extend([color]*len(coords))
        names.extend([cls_name]*len(coords))

    return points, colors, names


if __name__ == '__main__':
    # g = geojson.load(open2('/mnt/totem_data/totem/fengwentai/project/projectO/patch_cla_project/out_geojson_post/0507-MT03-P4-1-T1-1-20240429-S103187-9-OK.czi.json', 'r'))
    # conts, names = load_geojson('/mnt/totem_data/totem/fengwentai/project/projectO/patch_cla_project/out_geojson_post/0507-MT03-P4-1-T1-1-20240429-S103187-9-OK.czi.json')
    # save_geojson(conts, None, names, 'o.geojson')

    save_point_geojson(np.array([[1, 1], [20, 20], [300, 300]]), None, None, 'o.geojson')
    points = load_point_geojson('o.geojson')

    print(1)