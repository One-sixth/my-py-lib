import numpy as np
from scipy.special import softmax
from scipy.special import expit as sigmoid


def one_hot(class_array: np.ndarray, class_num, dim = -1, dtype = np.int32):
    '''
    可将[D1, D2, D3, ..., DN] 矩阵转换为 [D1, D2, ..., DN, D(N+1)] 的独热矩阵
    :param class_array: [D1, D2, ..., DN] 类别矩阵
    :param class_num:   类别数量
    :return:
    '''
    class_array = np.array(class_array, dtype=np.int32, copy=False)
    new_shape = [*([1]*class_array.ndim), class_num]
    a = np.arange(class_num).reshape(new_shape)
    b = np.equal(class_array[..., None], a).astype(dtype)
    if dim != -1:
        b = np.moveaxis(b, -1, dim)
    return b


def one_hot_invert(onehot_array, dim=-1):
    '''
    上面one_hot的逆操作
    可将[D1, D2, D3, ..., DN] 的独热矩阵转换为 [D1, D2, ..., D(N-1)] 的类别矩阵
    :param onehot_array: [D1, D2, ..., DN] 独热矩阵
    :param dim: 操作的维度，默认为-1
    :return: y => class array
    '''
    class_arr = np.argmax(onehot_array, axis=dim)
    return class_arr


# def softmax(x, axis=-1):
#     return np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)


# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))


def tr_figure_to_array(fig):
    '''
    转换 matplotlib 的 figure 到 numpy 数组
    '''
    fig.canvas.draw()
    mv = fig.canvas.buffer_rgba()
    im = np.asarray(mv)
    # 原图是 rgba，下面去除透明通道
    im = im[..., :3]
    # 需要复制，否则会映射到一个matlibplot的重用内存区，导致返回的图像会被破坏
    im = im.copy()
    return im


if __name__ == '__main__':
    class_num = 10
    a = np.random.randint(0, class_num, (3, 128, 128))
    h = one_hot(class_array=a, class_num=class_num)
    assert np.all(a == np.argmax(h, -1))

    import matplotlib.pyplot as plt
    import cv2

    figure = plt.figure()
    plot = figure.add_subplot(111)
    x = np.arange(1, 100, 0.1)
    y = np.sin(x) / x
    plot.plot(x, y)

    image = tr_figure_to_array(figure)
    cv2.imshow("tr_figure_to_array", image)
    cv2.waitKey(0)
