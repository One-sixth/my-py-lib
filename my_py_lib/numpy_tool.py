import numpy as np


def one_hot(class_array: np.ndarray, class_num):
    '''
    可将[D1, D2, D3, ..., DN] 矩阵转换为 [D1, D2, ..., DN, D(N+1)] 的独热矩阵
    :param class_array: [D1, D2, ..., DN] 类别矩阵
    :param class_num:   类别数量
    :return:
    '''
    class_array = np.array(class_array, copy=False)
    a = np.arange(class_num).reshape([*([1]*class_array.ndim), class_num])
    b = np.equal(class_array[..., None], a).astype(np.long)
    return b


if __name__ == '__main__':
    class_num = 10
    a = np.random.randint(0, class_num, (3, 128, 128))
    h = one_hot(class_array=a, class_num=class_num)
    assert np.all(a == np.argmax(h, -1))
