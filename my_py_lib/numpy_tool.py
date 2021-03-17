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


def softmax(x, axis=-1):
    return np.exp(x)/ np.sum(np.exp(x), axis=axis, keepdims=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tr_figure_to_array(fig):
    '''
    转换 matplotlib 的 figure 到 numpy 数组
    '''
    fig.canvas.draw()
    mv = fig.canvas.buffer_rgba()
    im = np.asarray(mv)
    # 原图是 rgba，下面去除透明通道
    im = im[..., :3]
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
