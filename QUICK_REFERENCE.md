# my-py-lib 快查文档

> **作者**: onesixth | **版本**: 0.0.1 | **平台**: Windows (主要)  
> **仓库**: https://github.com/One-sixth/my-py-lib  
> **定位**: 个人 Python 工具库，面向图像处理、计算机视觉、数据集处理、病理图分析

---

## 📁 项目目录结构

```
my-py-lib/
├── setup.py                        # 包安装配置
├── README.md                       # 项目说明
├── LICENSE                         # MIT 许可证
├── install.cmd / install.sh        # 安装脚本
├── QUICK_REFERENCE.md              # 本文件
└── my_py_lib/                      # 主包
    ├── __init__.py
    │
    │  ── 坐标与变换 ──
    ├── coord_tool.py               # 坐标格式转换 (xywh↔x1y1x2y2↔yxhw 等)
    ├── affine_matrix_tool.py       # 仿射矩阵生成 (旋转/平移/缩放/切变)
    ├── im_coords_affine_enhance_tool.py  # 图像+坐标联合仿射增强
    │
    │  ── 边界框 ──
    ├── bbox_tool.py                # 边界框操作 (面积/IoU/NMS/WBF/最短链接对)
    ├── bbox_eval_tool.py           # 边界框评估 (多阈值 IoU 评估)
    │
    │  ── 轮廓 ──
    ├── contour_tool.py             # 轮廓操作 (OpenCV+Shapely，格式转换/面积/IoU/Dice)
    ├── contour_eval_tool.py        # 轮廓评估 (IoU/Dice 多阈值评估)
    │
    │  ── 图像处理 ──
    ├── im_tool.py                  # 图像辅助 (resize/pad/维度检查/绘制)
    ├── color_enhance_tool.py       # 颜色增强 (HSV/对比度/亮度/随机裁剪翻转)
    ├── opsl_im_tool.py             # OpenSlide/TiffSlide 大图读取
    ├── heatmap_tool.py             # 热图工具 (检测点分类/FCOS tlbr 热图)
    ├── draw_tool.py                # 渐变圆绘制 (numba 加速)
    ├── draw_repel_code_tool.py     # Repel Code 绘制 (论文算法实现)
    │
    │  ── 评估系统 ──
    ├── score_tool.py               # 通用评分核心 (Fn/precision/recall)
    ├── class_eval_tool.py          # 分类评估
    ├── pixel_eval_tool.py          # 像素级评估
    ├── keypoint_eval_tool.py       # 关键点评估 (距离阈值匹配)
    │
    │  ── 点与数组 ──
    ├── point_tool.py               # 点操作 (最短链接对/仿射变换)
    ├── numpy_tool.py               # NumPy 工具 (one_hot/softmax/sigmoid/round_int)
    ├── list_tool.py                # 列表操作 (多取/多设/分组/布尔↔ID)
    │
    │  ── 大图处理 ──
    ├── multi_scale_large_image_result.py  # 多尺度大图结果聚合
    ├── multi_scale_patch_result.py        # 多尺度图块结果管理
    ├── ndarray_over_scan_wrapper.py       # 多维数组溢出采样
    ├── image_over_scan_wrapper.py         # 图像溢出范围采样
    ├── image_free_gauss_fusion_wrapper.py # 高斯融合大图拼接
    ├── coords_over_scan_gen.py            # 滑窗坐标生成器
    │
    │  ── 文件 IO ──
    ├── csv_tool.py                 # CSV 读写
    ├── json_tool.py                # JSON 读写
    ├── xlsx_tool.py                # Excel 读写
    ├── io_utils.py                 # 二进制 IO (配对 C++ 库)
    ├── path_tool.py                # 路径操作 (分割/查找/家目录)
    ├── download_tool.py            # 断点续传下载 (SHA1 校验)
    ├── simple_bucket.py            # 简单二进制桶存储
    ├── imagescope_xml_utils.py     # ImageScope XML 读写
    ├── asap_xml_utils.py           # ASAP XML 读写
    │
    │  ── 可视化与工具 ──
    ├── plot_tool.py                # matplotlib 绘图
    ├── mosaic_image.py             # 图像拼图 (自动布局)
    ├── auto_show_running.py        # 训练中实时图像显示
    ├── preload_generator.py        # 预加载生成器 (多线程)
    ├── universal_batch_generator.py # 通用批量生成器
    ├── qupath_label_tool.py        # QuPath GeoJSON 标注读写
    ├── tissue_tool.py              # 病理图组织轮廓提取
    ├── str_half2full_tool.py       # 全角半角转换
    ├── awkward_tool.py             # awkward 数组加载
    ├── utils.py                    # 杂项工具
    └── void_cls.py                 # 空类占位符
    │
    │  ── 数据集 ──
    └── dataset/
        ├── __init__.py
        ├── dataset.py              # 抽象数据集基类
        ├── voc_dataset.py          # VOC 2012 数据集
        ├── coco_dataset.py         # COCO 2017 数据集
        └── _cocostuffhelper.py     # COCO 分割辅助
```

---

## 📦 外部依赖

| 依赖 | 用途 | 必需模块 |
|------|------|---------|
| **numpy** | 核心数据结构 | 全部 |
| **opencv-python** (cv2) | 图像处理/绘制 | im_tool, contour_tool, draw_tool, color_enhance_tool 等 |
| **shapely** | 几何运算 (IoU/Dice/布尔运算) | contour_tool |
| **scikit-image** (skimage) | resize/disk | im_tool, heatmap_tool, draw_repel_code_tool |
| **scipy** | ndimage/softmax/sigmoid | im_tool, numpy_tool |
| **Pillow** (PIL) | 图像绘制/文字 | im_tool |
| **imageio** | 图像读写 | im_tool, image_free_gauss_fusion_wrapper |
| **lxml** | XML 解析 | asap_xml_utils |
| **matplotlib** | 绘图/颜色 | asap_xml_utils, plot_tool |
| **prettytable** | 表格格式化输出 | contour_eval_tool |
| **numba** | JIT 加速 | draw_tool |
| **requests** | HTTP 下载 | download_tool |
| **openpyxl** | Excel 读写 | xlsx_tool |
| **geojson** | GeoJSON 读写 | qupath_label_tool |
| **awkward** | 不规则数组 | awkward_tool |
| **pandas** | DataFrame 转换 | awkward_tool |
| **openslide** | 病理大图读取 | opsl_im_tool, multi_scale_* (可选) |
| **tiffslide** | TIFF 大图读取 | opsl_im_tool (可选，openslide 替代) |
| **pycocotools** | COCO 数据集 | coco_dataset |

---

## 🔧 核心模块速查

### coord_tool.py — 坐标格式转换

> 坐标系: x 为水平方向，y 为垂直方向

| 函数 | 说明 | 输入 → 输出 |
|------|------|------------|
| `xywh2yxhw(coord)` | xy 互换 + wh 互换 | xywh → yxhw |
| `xy2yx(coord)` | xy 互换 | xy → yx |
| `xywh_to_x1y1x2y2(coord)` | 中心+宽高 → 左上+右下 | xywh → x1y1x2y2 |
| `x1y1x2y2_to_xywh(coord)` | 左上+右下 → 中心+宽高 | x1y1x2y2 → xywh |
| `coord_pixelunit_to_scale(coord, shape)` | 像素坐标 → 归一化坐标 | 像素 → [0,1] |
| `coord_scale_to_pixelunit(coord, shape)` | 归一化坐标 → 像素坐标 | [0,1] → 像素 |

**别名**: `yxhw2xywh`, `x1y1x2y2_to_y1x1y2x2`, `y1x1y2x2_to_x1y1x2y2` 等均可互换使用。

---

### affine_matrix_tool.py — 仿射矩阵

> 坐标系: 窗口坐标系 (左上角原点，右为 X+，下为 Y+)  
> 矩阵用途: `new_xy = M @ old_xy`

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| `make_rotate(angle, center_yx, img_hw)` | 生成旋转矩阵 | angle: 顺时针角度; center_yx: 旋转中心 (百分比或绝对) |
| `make_move(move_yx, img_hw)` | 生成平移矩阵 | move_yx: 平移量 (百分比或像素) |
| `make_scale(scale_yx, center_yx, img_hw)` | 生成缩放矩阵 | scale_yx: 缩放倍率 (百分比) |
| `make_shear(shear_yx)` | 生成切变矩阵 | shear_yx: 切变角度 |

---

### bbox_tool.py — 边界框工具箱

> ⚠️ 所有 bbox 格式为 **y1x1y2x2** (左上角 y,x + 右下角 y,x)  
> 格式转换请先用 `coord_tool`

| 函数 | 说明 |
|------|------|
| `calc_bboxes_area(bbox)` | 计算面积 |
| `offset_bboxes(bbox, ori_yx, new_ori_yx)` | 重定位原点 |
| `resize_bboxes(bbox, factor_hw, center_yx)` | 缩放包围框 |
| `calc_bbox_center(bbox)` | 求中心点 |
| `inter_bbox_1to1(bbox1, bbox2)` | 求交集框 |
| `calc_bbox_iou_1toN(bbox1, bboxes)` | 1 对 N 的 IoU |
| `calc_bbox_iou_NtoM(bboxes1, bboxes2)` | N 对 M 的 IoU 矩阵 |
| `calc_bbox_occupancy_ratio_1toN(bbox1, bboxes)` | 1 对 N 占有率 |
| `pad_bbox_to_square(bbox)` | 填充为正方形 |
| `nms_process(confs, bboxes, iou_thresh)` | NMS 非极大值抑制 |
| `wbf_process(confs, bboxes, iou_thresh, conf_type)` | WBF 加权框融合 |
| `make_bbox_by_center_point(center_yx, bbox_hw)` | 中心点+尺寸 → bbox |
| `get_bboxes_shortest_link_pair(bboxes1, bboxes2, iou_th)` | 最短链接对匹配 |

---

### contour_tool.py — 轮廓工具箱

> 轮廓格式: OpenCV `[N,1,xy]`，本库 `[N,yx]`，Shapely `Polygon`  
> 优先用 OpenCV 函数（快），没有的才用 Shapely

| 函数 | 说明 |
|------|------|
| `tr_cv_to_my_contours(cv_contours)` | OpenCV → 本库格式 |
| `tr_my_to_cv_contours(my_contours)` | 本库 → OpenCV 格式 |
| `tr_my_to_polygon(my_contours)` | 本库 → Shapely Polygon |
| `calc_contour_area(contour)` | 求面积 |
| `calc_iou_with_contours_1toN(cont, conts)` | 1 对 N IoU |
| `calc_iou_with_contours_NtoM(conts1, conts2)` | N 对 M IoU 矩阵 |
| `calc_dice_contours(cont1, cont2)` | Dice 系数 |
| `find_contours(mask, mode, method)` | 从掩码提取轮廓 |
| `draw_contours(im, contours, ...)` | 绘制轮廓 |
| `resize_contours(contours, factor)` | 缩放轮廓 |
| `get_contours_shortest_link_pair(conts1, conts2, iou_th)` | 轮廓最短链接对 |

---

### im_tool.py — 图像辅助库

> 作为 scikit-image 和 OpenCV 的辅助，非替代

| 函数 | 说明 |
|------|------|
| `resize_image(img, target_hw, interpolation)` | 图像缩放 (支持 cv2/skimage) |
| `pad_picture(im, target_w, target_h, ...)` | 图像填充到指定尺寸 |
| `copy_make_border(im, top, bottom, left, right, value)` | 边界填充 |
| `ensure_image_has_3dim(im)` | 确保 3 维 (补通道维) |
| `ensure_image_has_same_ndim(im, ori_im)` | 与原图同维度 |
| `draw_bboxes(im, bboxes, ...)` | 绘制包围框 |
| `draw_texts(im, texts, positions, ...)` | 绘制文字 |
| `draw_keypoints(im, kps, ...)` | 绘制关键点 |
| `load_image(path)` | 加载图像 (imageio) |

---

### color_enhance_tool.py — 颜色增强

> 限定 RGB uint8 图像

| 函数 | 说明 |
|------|------|
| `random_adjust_HSV(img, h_range, s_range, v_range)` | 随机 HSV 调整 |
| `random_adjust_contrast_brightness(img, ...)` | 随机对比度+亮度 |
| `random_crop_and_resize(img, crop_range)` | 随机裁剪+缩放 |
| `random_flip(img, horizontal, vertical)` | 随机翻转 |
| `random_rotate90(img, times_range)` | 随机旋转 90° |

---

### 评估系统

#### score_tool.py — 评分核心

| 函数 | 说明 |
|------|------|
| `calc_score_fn(prec, recall, n)` | 计算 Fn 分数 (n=0.5/1/2) |
| `calc_score_prec_recall(n_label_found, ...)` | 计算 precision 和 recall |
| `calc_score_f05_f1_f2_prec_recall(...)` | 聚合计算全部分数 |

#### bbox_eval_tool.py / contour_eval_tool.py / keypoint_eval_tool.py / pixel_eval_tool.py / class_eval_tool.py

统一评估接口模式:
```python
# 通用评估调用
scores = calc_xxx_score(preds, pred_classes, labels, label_classes, 
                        classes_list, match_thresh_list)
# 汇总
summary = summary_xxx_score(scores, cls_list, match_thresh_list)
```

---

### 数据集

#### dataset.py — 抽象基类

```python
class Dataset:
    get_label_num()          # 标签数量
    get_class_num()          # 类别数量
    get_class_name()         # 类别名称列表
    shuffle()                # 打乱
    get_label_info(id)       # 标签详情 dict
    get_label_image(id)      # 原图
    get_label_instance_bbox(id)      # bbox (目标检测)
    get_label_class_mask(id)         # 语义分割掩码
    get_label_instance_mask(id)      # 实例分割掩码
    get_label_instance_keypoints(id) # 关键点
```

#### voc_dataset.py

```python
from my_py_lib.dataset.voc_dataset import VocDataset
ds = VocDataset('path/to/VOC2012', 'train')
img = ds.get_label_image(0)
bbox, cls = ds.get_label_instance_bbox(0)
```

#### coco_dataset.py

```python
from my_py_lib.dataset.coco_dataset import CocoDataset
ds = CocoDataset('train2017', 'instances', 'path/to/coco')
img = ds.get_label_image(0)
bbox, cls = ds.get_label_instance_bbox(0)
mask = ds.get_label_class_mask(0)
kps = ds.get_label_instance_keypoints(0)
```

---

### 大图处理工具

#### opsl_im_tool.py — OpenSlide/TiffSlide 工具

| 函数 | 说明 |
|------|------|
| `read_region_any_ds(opsl_im, ds_factor, start_yx, region_hw)` | 任意下采样读取 |
| `make_thumb_any_level(opsl_im, ds_factor)` | 任意级别缩略图 |
| `get_level0_mpp(opsl_im)` | 获取 0 级 MPP (微米/像素) |

#### coords_over_scan_gen.py — 滑窗坐标生成

```python
from my_py_lib.coords_over_scan_gen import n_step_scan_coords_gen_v2
for yx_start, yx_end in n_step_scan_coords_gen_v2(im_hw=(5120,5120), window_hw=(512,512), n_step=0.5):
    patch = img[yx_start[0]:yx_end[0], yx_start[1]:yx_end[1]]
```

#### image_over_scan_wrapper.py — 溢出采样

```python
wrapper = ImageOverScanWrapper(img)
patch = wrapper.get(yx_start=(-100,-100), yx_end=(100,100), pad_value=0)
# 坐标超出边界时自动填充
```

#### multi_scale_large_image_result.py — 大图结果聚合

```python
mslir = MultiScaleLargeImageResult(opsl_im, n_class=3, 
                                    level_0_big_patch_hw=(5120,5120),
                                    ds_factors=(1, 2, 4))
# 生成图块 → 处理 → 合并结果
```

---

### 文件 IO 工具

| 模块 | 函数 | 说明 |
|------|------|------|
| csv_tool | `load_csv(file)` / `save_csv(rows, file)` | CSV 读写 |
| json_tool | `load_json(path)` / `save_json(obj, path)` | JSON 读写 |
| xlsx_tool | `load_xlsx(file, sheet)` / `save_xlsx(rows, file)` | Excel 读写 |
| path_tool | `split_file_path(p)` / `find_file_by_exts(dir, exts)` | 路径操作 |
| download_tool | `download_file(url, fp, sha1_code)` | 断点续传下载 |
| io_utils | `ReadType(f, T)` / `WriteType(f, data)` / `ReadVector(f, T)` | 二进制 IO |
| simple_bucket | `SimpleBucketWriter(file)` / `SimpleBucketReader(file)` | 桶存储 |

---

### 其他工具

| 模块 | 说明 |
|------|------|
| plot_tool | matplotlib 绘图: `plot_scatter_2d`, `plot_bar_2d` |
| mosaic_image | 图像拼图: `create_mosaic(images, layout=(rows,cols))` |
| auto_show_running | 训练实时显示: `AutoShowRunning(out_hw, show_num_hw)` |
| preload_generator | 多线程预加载: `preload_generator(g, queue_size=10)` |
| universal_batch_generator | 批量打包: `universal_batch_generator(g, batch_size)` |
| tissue_tool | 组织轮廓提取: `get_tissue_contours(im, gray_thresh=210)` |
| qupath_label_tool | QuPath GeoJSON: `save_line_string_geojson(...)` / `load_line_string_geojson(...)` |
| str_half2full_tool | 全角半角: `str_full2half(s)` / `str_half2full(s)` |
| ndarray_over_scan_wrapper | 数组溢出采样: `NdArrayOverScanWrapper(arr)` |
| awkward_tool | awkward 加载: `load_ak_from_csv(file)` / `load_ak_from_excel(file)` |

---

## 🎯 常见使用场景

### 场景 1: 目标检测评估

```python
from my_py_lib.bbox_tool import calc_bbox_iou_NtoM, nms_process
from my_py_lib.bbox_eval_tool import calc_bbox_score, summary_bbox_score
from my_py_lib.coord_tool import xywh_to_x1y1x2y2

# 坐标转换
bboxes = xywh_to_x1y1x2y2(raw_bboxes)

# NMS 过滤
keep = nms_process(confs, bboxes, iou_thresh=0.5)

# 评估
scores = calc_bbox_score(pred_bboxes, pred_classes, label_bboxes, label_classes, 
                         classes_list=[0,1,2], match_iou_thresh_list=(0.3, 0.5, 0.7))
summary = summary_bbox_score(scores, cls_list=[0,1,2], match_iou_thresh_list=(0.3, 0.5, 0.7))
```

### 场景 2: 语义分割评估

```python
from my_py_lib.contour_tool import find_contours, calc_iou_with_contours_NtoM
from my_py_lib.contour_eval_tool import calc_contour_score, summary_contour_score

# 从掩码提取轮廓
pred_contours = find_contours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 评估
scores = calc_contour_score(pred_contours, pred_classes, label_contours, label_classes,
                            classes_list=[0,1], match_iou_thresh_list=(0.3, 0.5, 0.7))
```

### 场景 3: 数据增强

```python
from my_py_lib.im_coords_affine_enhance_tool import random_apply_affine_to_img_and_bboxes
from my_py_lib.color_enhance_tool import random_adjust_HSV, random_flip

# 颜色增强
img = random_adjust_HSV(img, h_range=0.1, s_range=0.2, v_range=0.2)
img = random_flip(img, horizontal=True)

# 仿射增强 (图像+坐标联合变换)
img, bboxes = random_apply_affine_to_img_and_bboxes(img, bboxes, ...)
```

### 场景 4: 病理大图处理

```python
import openslide
from my_py_lib.opsl_im_tool import read_region_any_ds, make_thumb_any_level
from my_py_lib.tissue_tool import get_tissue_contours
from my_py_lib.multi_scale_large_image_result import MultiScaleLargeImageResult

# 打开大图
opsl_im = openslide.OpenSlide('path/to/slide.ndpi')

# 获取组织轮廓
thumb = make_thumb_any_level(opsl_im, ds_factor=32)
tissue_contours = get_tissue_contours(thumb)

# 多尺度大图处理
mslir = MultiScaleLargeImageResult(opsl_im, n_class=3, ...)
```

### 场景 5: 图像拼图展示

```python
from my_py_lib.mosaic_image import create_mosaic

images = [img1, img2, img3, img4, img5, img6]
mosaic = create_mosaic(images, layout=(2, 3), padding=5, background_color=128)
```

---

## ⚠️ 坐标格式约定

本库中存在多种坐标格式，使用时务必注意:

| 格式 | 含义 | 示例 |
|------|------|------|
| **xywh** | x_center, y_center, width, height | `[100, 200, 50, 80]` |
| **x1y1x2y2** | 左上角 (x,y) + 右下角 (x,y) | `[75, 160, 125, 240]` |
| **y1x1y2x2** | 左上角 (y,x) + 右下角 (y,x) ⭐ bbox_tool 默认 | `[160, 75, 240, 125]` |
| **yxhw** | y, x, height, width | `[160, 75, 80, 50]` |
| **yx** | y, x 坐标点 | `[160, 75]` |
| **xy** | x, y 坐标点 | `[75, 160]` |

> ⭐ **bbox_tool 和 contour_tool 默认使用 y1x1y2x2 / yx 格式**  
> 如需其他格式，先用 `coord_tool` 转换

---

## 🔗 模块依赖关系

```
score_tool ←── bbox_eval_tool
           ←── contour_eval_tool
           ←── class_eval_tool
           ←── pixel_eval_tool
           ←── keypoint_eval_tool

affine_matrix_tool ←── point_tool
                   ←── im_coords_affine_enhance_tool

point_tool ←── keypoint_eval_tool
           ←── contour_tool

bbox_tool ←── bbox_eval_tool
          ←── heatmap_tool
          ←── image_free_gauss_fusion_wrapper

contour_tool ←── contour_eval_tool
             ←── multi_scale_patch_result
             ←── tissue_tool

im_tool ←── image_over_scan_wrapper
        ←── auto_show_running
        ←── contour_tool
        ←── im_coords_affine_enhance_tool

list_tool ←── bbox_eval_tool
          ←── contour_eval_tool
          ←── image_free_gauss_fusion_wrapper
```

---

## 📝 安装

```bash
# 方式 1: pip 安装
pip install -e .

# 方式 2: 脚本安装
# Windows
install.cmd
# Linux
bash install.sh
```

---

## 💡 设计注意事项

1. **坐标格式统一**: 调用前确认函数期望的坐标格式，必要时用 `coord_tool` 转换
2. **轮廓格式**: OpenCV `[N,1,xy]`，本库 `[N,yx]`，Shapely `Polygon` — 用 `tr_*` 函数互转
3. **dtype 要求**: 大多数函数要求 `np.float32` 输入，整数坐标用 `np.int32`
4. **维度约定**: 图像 `[H,W,C]`，bbox `[y1,x1,y2,x2]`，点 `[y,x]`
5. **OpenCV 版本**: 要求 `>= 4.0`
6. **numba 版本**: 要求 `>= 0.48.0`

---

*最后更新: 2026-05-29*
