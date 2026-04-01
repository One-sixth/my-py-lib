
"""
图像拼图工具 - 自动将多个numpy图像拼接成自然的拼图布局

功能特点:
- 支持混合单通道(灰度)和三通道(RGB)图像输入
- 支持图像大小不一致的情况，自动智能调整
- 自动计算最佳行列布局，使拼图接近正方形
- 输出尺寸约束: 500x500 ~ 10000x10000
- 简单灵活的API接口

作者: Qingyan Agent
"""

import numpy as np
from typing import List, Union, Tuple, Optional
import math
import cv2


def create_mosaic(
        images: List[np.ndarray],
        target_size: Optional[Tuple[int, int]] = None,
        padding: int = 2,
        background_color: Union[int, Tuple[int, int, int]] = 0,
        maintain_aspect: bool = True,
        force_rgb: bool = True,
        layout: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    将多个numpy图像自动拼接成一个自然的拼图布局。

    参数:
    ----------
    images : List[np.ndarray]
        图像列表，每个图像可以是单通道(H,W)或三通道(H,W,3)的numpy数组
        支持混合不同通道数和不同尺寸的图像

    target_size : Optional[Tuple[int, int]], 默认 None
        目标输出尺寸 (height, width)
        如果为None，则自动计算合适的尺寸
        最终输出会约束在 500x500 ~ 10000x10000 范围内

    padding : int, 默认 2
        图像之间的间距像素数

    background_color : Union[int, Tuple[int, int, int]], 默认 0
        背景颜色，单通道图像用int，三通道图像用tuple
        默认为黑色(0 或 (0,0,0))

    maintain_aspect : bool, 默认 True
        是否保持图像原始宽高比
        True: 保持宽高比，可能添加黑边
        False: 拉伸图像以填充单元格

    force_rgb : bool, 默认 True
        是否强制转换为RGB格式输出
        True: 所有图像转换为RGB，输出三通道图像
        False: 如果所有输入都是灰度图，输出灰度图

    layout : Optional[Tuple[int, int]], 默认 None
        自定义布局 (rows, cols)
        如果为None，则自动计算最佳布局（接近正方形）
        如果指定，则使用指定的行列数

    返回:
    ----------
    np.ndarray
        拼接后的图像数组，形状为 (H, W) 或 (H, W, 3)

    示例:
    ----------
    >>> import numpy as np
    >>> # 创建一些测试图像
    >>> gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    >>> rgb_img = np.random.randint(0, 255, (150, 200, 3), dtype=np.uint8)
    >>> # 拼接图像
    >>> mosaic = create_mosaic([gray_img, rgb_img, gray_img, rgb_img])
    >>> print(mosaic.shape)
    >>> # 指定布局
    >>> mosaic = create_mosaic([gray_img, rgb_img, gray_img, rgb_img], layout=(2, 2))
    >>> print(mosaic.shape)
    """

    # 参数验证
    if len(images) == 0:
        raise ValueError("图像列表不能为空")

    if not all(isinstance(img, np.ndarray) for img in images):
        raise TypeError("所有图像必须是numpy数组")

    # 预处理图像：统一通道格式
    processed_images = _preprocess_images(images, force_rgb)

    # 确定输出通道数
    is_rgb = force_rgb or any(_is_rgb_image(img) for img in images)

    # 计算布局
    num_images = len(processed_images)
    if layout is not None:
        rows, cols = layout
    else:
        rows, cols = _calculate_optimal_layout(num_images)

    # 计算单元格尺寸
    cell_height, cell_width = _calculate_cell_size(
        processed_images, rows, cols, target_size, padding
    )

    # 调整图像尺寸并放置到网格中
    mosaic = _assemble_mosaic(
        processed_images,
        rows, cols,
        cell_height, cell_width,
        padding, background_color, maintain_aspect, is_rgb
    )

    # 应用尺寸约束
    mosaic = _apply_size_constraints(mosaic)

    return mosaic


def _is_rgb_image(img: np.ndarray) -> bool:
    """判断图像是否为RGB格式"""
    return len(img.shape) == 3 and img.shape[2] == 3


def _preprocess_images(images: List[np.ndarray], force_rgb: bool) -> List[np.ndarray]:
    """
    预处理图像：统一数据类型和通道格式
    """
    processed = []

    for img in images:
        # 确保是uint8类型
        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)

        # 处理不同维度的输入
        if len(img.shape) == 2:
            # 单通道图像
            if force_rgb:
                # 转换为RGB
                img = np.stack([img] * 3, axis=-1)
        elif len(img.shape) == 3:
            if img.shape[2] == 1:
                # (H, W, 1) 格式
                if force_rgb:
                    img = np.concatenate([img] * 3, axis=-1)
                else:
                    img = img.squeeze(-1)
            elif img.shape[2] == 4:
                # RGBA格式，转换为RGB
                img = img[:, :, :3]
            # 其他情况保持不变
        else:
            raise ValueError(f"不支持的图像维度: {img.shape}")

        processed.append(img)

    return processed


def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """将图像归一化为uint8类型"""
    img = img.astype(np.float64)
    img_min, img_max = img.min(), img.max()

    if img_max - img_min > 0:
        img = (img - img_min) / (img_max - img_min) * 255
    else:
        img = np.zeros_like(img)

    return img.astype(np.uint8)


def _calculate_optimal_layout(num_images: int) -> Tuple[int, int]:
    """
    计算最佳的行列布局，使拼图尽可能接近正方形

    策略：找到最接近正方形的布局，同时保证能容纳所有图像
    """
    if num_images <= 0:
        return 1, 1

    # 计算最接近正方形的布局
    sqrt_n = math.sqrt(num_images)

    # 尝试找到最佳的行列组合
    best_diff = float('inf')
    best_rows, best_cols = 1, num_images

    for cols in range(1, num_images + 1):
        rows = math.ceil(num_images / cols)
        # 计算与正方形的差异
        diff = abs(rows - cols) + abs(rows * cols - num_images) * 0.1

        if diff < best_diff:
            best_diff = diff
            best_rows, best_cols = rows, cols

    return best_rows, best_cols


def _calculate_cell_size(
        images: List[np.ndarray],
        rows: int,
        cols: int,
        target_size: Optional[Tuple[int, int]],
        padding: int
) -> Tuple[int, int]:
    """
    计算每个单元格的尺寸
    """
    # 找出所有图像的最大高度和宽度
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)

    if target_size is not None:
        # 根据目标尺寸计算单元格大小
        target_height, target_width = target_size
        cell_height = (target_height - padding * (rows + 1)) // rows
        cell_width = (target_width - padding * (cols + 1)) // cols
    else:
        # 使用最大尺寸作为基准，并适当放大以适应布局
        # 考虑整体宽高比
        aspect_ratio = (cols * max_width) / (rows * max_height) if rows > 0 else 1

        if aspect_ratio > 1.5:
            # 太宽，增加高度
            cell_height = max_height
            cell_width = max_width
        elif aspect_ratio < 0.67:
            # 太高，增加宽度
            cell_height = max_height
            cell_width = max_width
        else:
            # 比较均衡
            cell_height = max_height
            cell_width = max_width

    # 确保最小尺寸
    cell_height = max(cell_height, 50)
    cell_width = max(cell_width, 50)

    return cell_height, cell_width


def _assemble_mosaic(
        images: List[np.ndarray],
        rows: int,
        cols: int,
        cell_height: int,
        cell_width: int,
        padding: int,
        background_color: Union[int, Tuple[int, int, int]],
        maintain_aspect: bool,
        is_rgb: bool
) -> np.ndarray:
    """
    将图像组装成拼图
    """
    # 计算输出图像尺寸
    out_height = rows * cell_height + padding * (rows + 1)
    out_width = cols * cell_width + padding * (cols + 1)

    # 创建输出画布
    if is_rgb:
        if isinstance(background_color, int):
            bg = (background_color,) * 3
        else:
            bg = background_color
        mosaic = np.full((out_height, out_width, 3), bg, dtype=np.uint8)
    else:
        bg = background_color if isinstance(background_color, int) else background_color[0]
        mosaic = np.full((out_height, out_width), bg, dtype=np.uint8)

    # 放置每个图像
    for idx, img in enumerate(images):
        if idx >= rows * cols:
            break

        row = idx // cols
        col = idx % cols

        # 计算单元格位置
        y_start = padding + row * (cell_height + padding)
        x_start = padding + col * (cell_width + padding)

        # 调整图像尺寸
        resized = _resize_image(img, cell_height, cell_width, maintain_aspect, is_rgb)

        # 计算居中位置
        img_h, img_w = resized.shape[:2]
        y_offset = (cell_height - img_h) // 2
        x_offset = (cell_width - img_w) // 2

        # 放置图像
        y_end = y_start + y_offset + img_h
        x_end = x_start + x_offset + img_w

        if is_rgb:
            mosaic[y_start + y_offset:y_end, x_start + x_offset:x_end] = resized
        else:
            mosaic[y_start + y_offset:y_end, x_start + x_offset:x_end] = resized

    return mosaic


def _resize_image(
        img: np.ndarray,
        target_height: int,
        target_width: int,
        maintain_aspect: bool,
        is_rgb: bool
) -> np.ndarray:
    """
    调整图像尺寸
    """
    h, w = img.shape[:2]

    if maintain_aspect:
        # 保持宽高比，计算缩放比例
        scale = min(target_height / h, target_width / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
    else:
        # 直接拉伸
        new_h, new_w = target_height, target_width

    # 使用简单的最近邻插值（不依赖cv2）
    resized = _nearest_neighbor_resize(img, new_h, new_w)

    return resized


# def _nearest_neighbor_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
#     """
#     最近邻插值缩放图像（不依赖外部库）
#     """
#     h, w = img.shape[:2]
#
#     if len(img.shape) == 3:
#         # RGB图像
#         resized = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
#         for i in range(new_h):
#             for j in range(new_w):
#                 src_i = int(i * h / new_h)
#                 src_j = int(j * w / new_w)
#                 src_i = min(src_i, h - 1)
#                 src_j = min(src_j, w - 1)
#                 resized[i, j] = img[src_i, src_j]
#     else:
#         # 灰度图像
#         resized = np.zeros((new_h, new_w), dtype=img.dtype)
#         for i in range(new_h):
#             for j in range(new_w):
#                 src_i = int(i * h / new_h)
#                 src_j = int(j * w / new_w)
#                 src_i = min(src_i, h - 1)
#                 src_j = min(src_j, w - 1)
#                 resized[i, j] = img[src_i, src_j]
#
#     return resized


def _nearest_neighbor_resize(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """
    最近邻插值缩放图像（不依赖外部库）
    """
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        resized = np.zeros((new_h, new_w, img.shape[2]), dtype=img.dtype)
    else:
        resized = np.zeros((new_h, new_w), dtype=img.dtype)
    resized[:] = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return resized


def _apply_size_constraints(mosaic: np.ndarray) -> np.ndarray:
    """
    应用尺寸约束：500x500 ~ 10000x10000
    """
    h, w = mosaic.shape[:2]
    min_size = 500
    # max_size = 10000
    max_size = 8192

    # 检查是否需要放大
    if h < min_size or w < min_size:
        scale = max(min_size / h, min_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        mosaic = _nearest_neighbor_resize(mosaic, new_h, new_w)

    # 检查是否需要缩小
    h, w = mosaic.shape[:2]
    if h > max_size or w > max_size:
        scale = min(max_size / h, max_size / w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        mosaic = _nearest_neighbor_resize(mosaic, new_h, new_w)

    return mosaic


# ============ 便捷函数 ============

def create_grid_mosaic(
        images: List[np.ndarray],
        rows: int,
        cols: int,
        **kwargs
) -> np.ndarray:
    """
    创建指定行列数的网格拼图

    参数:
    ----------
    images : List[np.ndarray]
        图像列表
    rows : int
        行数
    cols : int
        列数
    **kwargs :
        其他参数传递给 create_mosaic

    返回:
    ----------
    np.ndarray
        拼接后的图像
    """
    # 填充或截断图像列表以匹配网格大小
    images = list(images)  # 创建副本，避免修改原列表
    total_cells = rows * cols
    if len(images) < total_cells:
        # 用黑色图像填充
        sample_img = images[0] if images else np.zeros((100, 100), dtype=np.uint8)
        if len(sample_img.shape) == 2:
            filler = np.zeros((sample_img.shape[0], sample_img.shape[1]), dtype=np.uint8)
        else:
            filler = np.zeros((sample_img.shape[0], sample_img.shape[1], 3), dtype=np.uint8)

        while len(images) < total_cells:
            images.append(filler.copy())
    elif len(images) > total_cells:
        images = images[:total_cells]

    # 使用自定义布局
    result = create_mosaic(images, layout=(rows, cols), **kwargs)
    return result


def create_row_mosaic(images: List[np.ndarray], **kwargs) -> np.ndarray:
    """
    创建单行水平拼图

    参数:
    ----------
    images : List[np.ndarray]
        图像列表
    **kwargs :
        其他参数传递给 create_mosaic

    返回:
    ----------
    np.ndarray
        单行拼接的图像
    """
    return create_grid_mosaic(images, rows=1, cols=len(images), **kwargs)


def create_column_mosaic(images: List[np.ndarray], **kwargs) -> np.ndarray:
    """
    创建单列垂直拼图

    参数:
    ----------
    images : List[np.ndarray]
        图像列表
    **kwargs :
        其他参数传递给 create_mosaic

    返回:
    ----------
    np.ndarray
        单列拼接的图像
    """
    return create_grid_mosaic(images, rows=len(images), cols=1, **kwargs)


# ============ 测试代码 ============

if __name__ == "__main__":
    print("=" * 60)
    print("图像拼图工具测试")
    print("=" * 60)

    # 创建测试图像
    np.random.seed(42)

    # 测试1: 混合灰度和RGB图像
    print("\n测试1: 混合灰度和RGB图像")
    gray_small = np.random.randint(0, 255, (80, 100), dtype=np.uint8)
    gray_large = np.random.randint(0, 255, (150, 120), dtype=np.uint8)
    rgb_small = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
    rgb_large = np.random.randint(0, 255, (200, 180, 3), dtype=np.uint8)

    images_mixed = [gray_small, rgb_small, gray_large, rgb_large]
    mosaic1 = create_mosaic(images_mixed)
    print(f"  输入: 4张混合图像 (灰度+RGB, 不同尺寸)")
    print(f"  输出形状: {mosaic1.shape}")

    # 测试2: 纯灰度图像
    print("\n测试2: 纯灰度图像")
    gray_images = [np.random.randint(0, 255, (100, 100), dtype=np.uint8) for _ in range(6)]
    mosaic2 = create_mosaic(gray_images, force_rgb=False)
    print(f"  输入: 6张灰度图像")
    print(f"  输出形状: {mosaic2.shape}")

    # 测试3: 指定目标尺寸
    print("\n测试3: 指定目标尺寸")
    mosaic3 = create_mosaic(images_mixed, target_size=(800, 800))
    print(f"  输入: 4张混合图像, 目标尺寸: (800, 800)")
    print(f"  输出形状: {mosaic3.shape}")

    # 测试4: 单行拼图
    print("\n测试4: 单行拼图")
    mosaic4 = create_row_mosaic(images_mixed)
    print(f"  输入: 4张混合图像, 单行布局")
    print(f"  输出形状: {mosaic4.shape}")
    h, w = mosaic4.shape[:2]
    print(f"  宽高比: {w / h:.2f} (应该较宽)")

    # 测试5: 单列拼图
    print("\n测试5: 单列拼图")
    mosaic5 = create_column_mosaic(images_mixed)
    print(f"  输入: 4张混合图像, 单列布局")
    print(f"  输出形状: {mosaic5.shape}")
    h, w = mosaic5.shape[:2]
    print(f"  宽高比: {w / h:.2f} (应该较窄)")

    # 测试6: 自定义布局
    print("\n测试6: 自定义布局 (2x3)")
    mosaic6 = create_mosaic(images_mixed + [gray_small, gray_large], layout=(2, 3))
    print(f"  输入: 6张图像, 布局: 2行3列")
    print(f"  输出形状: {mosaic6.shape}")

    # 测试7: 大量图像
    print("\n测试7: 大量图像 (20张)")
    many_images = [np.random.randint(0, 255, (50 + i * 5, 60 + i * 3, 3), dtype=np.uint8) for i in range(20)]
    mosaic7 = create_mosaic(many_images)
    print(f"  输入: 20张不同尺寸的RGB图像")
    print(f"  输出形状: {mosaic7.shape}")

    # 测试8: 尺寸约束测试
    print("\n测试8: 尺寸约束测试")
    tiny_images = [np.random.randint(0, 255, (10, 10), dtype=np.uint8) for _ in range(4)]
    mosaic8 = create_mosaic(tiny_images)
    print(f"  输入: 4张极小图像 (10x10)")
    print(f"  输出形状: {mosaic8.shape} (应满足最小500x500约束)")

    # 测试9: 不同间距和背景色
    print("\n测试9: 自定义间距和背景色")
    mosaic9 = create_mosaic(images_mixed, padding=10, background_color=(50, 50, 100))
    print(f"  输入: 4张图像, 间距=10, 背景色=(50,50,100)")
    print(f"  输出形状: {mosaic9.shape}")

    # 测试10: 不保持宽高比
    print("\n测试10: 不保持宽高比 (拉伸填充)")
    mosaic10 = create_mosaic(images_mixed, maintain_aspect=False)
    print(f"  输入: 4张图像, maintain_aspect=False")
    print(f"  输出形状: {mosaic10.shape}")

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)

    # 保存测试结果
    try:
        from PIL import Image
        import os
        out_dir = "./results_test"
        os.makedirs(out_dir, exist_ok=True)

        # 保存多个示例结果
        result_img = Image.fromarray(mosaic1)
        result_img.save(f"{out_dir}/mosaic_test_result.png")
        print(f"\n示例结果已保存到: {out_dir}/mosaic_test_result.png")

        # 保存单行和单列结果
        Image.fromarray(mosaic4).save(f"{out_dir}/mosaic_row.png")
        Image.fromarray(mosaic5).save(f"{out_dir}/mosaic_column.png")
        print(f"单行拼图: {out_dir}/mosaic_row.png")
        print(f"单列拼图: {out_dir}/mosaic_column.png")
    except ImportError:
        print("\n提示: 安装Pillow可保存图像文件 (pip install Pillow)")
