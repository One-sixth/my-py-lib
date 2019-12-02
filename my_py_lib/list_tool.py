from typing import Iterable


def list_multi_get_with_ids(self: list, ids: Iterable):
    return [self[i] for i in ids]


def list_multi_get_with_bool(self: list, bools: Iterable):
    assert len(self) == len(bools)
    a = [self[i] for i, b in enumerate(bools) if b]
    return a


def list_multi_set_with_ids(self: list, ids: Iterable, items: Iterable):
    assert len(ids) == len(items)
    for _id, item in zip(ids, items):
        self[_id] = item


def list_multi_set_with_bool(self: list, bools: Iterable, items: Iterable):
    assert len(self) == len(bools)
    wait_set_ids = []
    for i, b in enumerate(bools):
        if b:
            wait_set_ids.append(i)
    assert len(wait_set_ids) == len(items)
    for i, item in zip(wait_set_ids, items):
        self[i] = item


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6]
    b = list_multi_get_with_ids(a, [0, 2, 4])
    assert a[0] == b[0] and a[2] == b[1] and a[4] == b[2]

    c = list_multi_get_with_bool(a, [True, False, True, False, False, False])
    assert a[0] == c[0] and a[2] == c[1]

    list_multi_set_with_ids(a, [0, 2], [3, 1])
    assert a[0] == 3 and a[2] == 1

    list_multi_set_with_bool(a, [True, False, True, False, False, False], [1, 3])
    assert a[0] == 1 and a[2] == 3
