'''
openslide image tools
'''

import openslide as opsl
import numpy as np
import cv2
import my_py_lib.im_tool as im_tool


def opsl_read_region_any_ds(opsl_im: opsl.OpenSlide, ds_factor, level_0_start_yx, level_0_region_hw, close_thresh=0.0):
    '''
    从多分辨率图中读取任意尺度图像
    :param opsl_im:             待读取的 OpenSlide 图像
    :param ds_factor:           下采样尺度
    :param level_0_start_yx:    所读取区域在尺度0上的位置
    :param level_0_region_hw:   所读取区域在尺度0上的高宽
    :param close_thresh:        如果找到足够接近的下采样尺度，则直接使用最接近的，不再对图像进行缩放，默认为0，即为关闭
    :return:
    '''
    level_downsamples = opsl_im.level_downsamples

    assert ds_factor > 0, f'Error! Not allow ds_factor <= 0. ds_factor={ds_factor}'

    # base_level = None
    # ori_patch_hw = None
    target_patch_hw = np.array(level_0_region_hw, np.int) // ds_factor

    is_close_list = np.isclose(ds_factor, level_downsamples, rtol=close_thresh, atol=0)
    if np.any(is_close_list):
        # 如果有足够接近的
        level = np.argmax(is_close_list)
        base_level = level
        ori_patch_hw = target_patch_hw
    else:
        # 没有足够接近的，则寻找最接近，并且分辨率更高的，然后再缩放。
        # 增加ds_factor超过level_downsamples边界的支持
        if ds_factor > max(opsl_im.level_downsamples):
            # 如果ds_factor大于图像自身包含的最大的倍率
            level = opsl_im.level_count - 1
        elif ds_factor < min(opsl_im.level_downsamples):
            # 如果ds_factor小于图像自身最小的倍率
            level = 0
        else:
            level = np.argmax(ds_factor < np.array(opsl_im.level_downsamples)) - 1
        assert level >= 0, 'Error! read_im_mod found unknow level {}'.format(level)
        base_level = level
        level_ds_factor = level_downsamples[level]
        ori_patch_hw = np.array(target_patch_hw / level_ds_factor * ds_factor, np.int)

    # 读取图块，如果不是目标大小则缩放到目标大小
    im = np.array(opsl_im.read_region(level_0_start_yx[::-1], base_level, ori_patch_hw[::-1]), np.uint8)[:, :, :3]
    if np.any(ori_patch_hw != target_patch_hw):
        im = im_tool.resize_image(im, target_patch_hw, cv2.INTER_AREA)
    return im


if __name__ == '__main__':
    import imageio

    opsl_im = opsl.OpenSlide(r"#142.ndpi")
    im = opsl_read_region_any_ds(opsl_im, ds_factor=4, level_0_start_yx = (20000, 20000), level_0_region_hw = (5120, 5120))
    imageio.imwrite('1.jpg', im)
