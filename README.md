# my_py_lib

个人 Python 工具集，主要面向**计算机视觉**和**数字病理图像处理**场景。

> ⚠️ 仅在 Windows 平台测试过，代码可能存在不规范之处，使用前请自行检查。

---

## 模块一览

### 📐 坐标与几何

| 模块 | 说明 |
|------|------|
| `coord_tool.py` | 坐标格式互转（xywh / yxhw / x1y1x2y2 等） |
| `bbox_tool.py` | 包围框操作：面积、IoU、NMS、WBF、最短配对等 |
| `contour_tool.py` | 轮廓操作：格式转换、面积、IoU、偏移、缩放、绘制等（基于 OpenCV + Shapely） |
| `point_tool.py` | 点集操作：最短配对、仿射变换、缩放、包含检测 |
| `affine_matrix_tool.py` | 仿射矩阵生成：旋转、平移、缩放、切变 |
| `heatmap_tool.py` | 热图生成与解析：TLBR 热图（FCOS 风格）、分类热图轮廓提取 |

### 🖼️ 图像处理

| 模块 | 说明 |
|------|------|
| `im_tool.py` | 图像工具集：缩放、填充、裁剪、文字绘制、热图叠加、WebP 编码等 |
| `opsl_im_tool.py` | OpenSlide / TiffSlide 大图读取：任意 MPP 缩放、缩略图生成 |
| `color_enhance_tool.py` | 图像增强：HSV 调整、对比度亮度、椒盐噪声、高斯噪声 |
| `draw_tool.py` | 渐变圆绘制（numba 加速） |
| `draw_repel_code_tool.py` | 反斥编码图绘制，用于细胞检测（论文: Enhanced Center Coding） |
| `auto_show_running.py` | 训练时实时显示图像网格 |
| `mosaic_image.py` | 图像拼图：自动布局、支持混合通道和尺寸 |

### 📊 评估工具

| 模块 | 说明 |
|------|------|
| `score_tool.py` | 精确率、召回率、F0.5 / F1 / F2 计算 |
| `bbox_eval_tool.py` | 包围框评估：多 IoU 阈值、多类别统计 |
| `contour_eval_tool.py` | 轮廓评估：IoU、Dice、面积统计 |
| `pixel_eval_tool.py` | 像素级评估 |
| `class_eval_tool.py` | 多分类评估 |
| `keypoint_eval_tool.py` | 关键点评估：多距离阈值 |

### 🔄 大图扫描与融合

| 模块 | 说明 |
|------|------|
| `coords_over_scan_gen.py` | N 分之一步长滑动窗口坐标生成器 |
| `image_over_scan_wrapper.py` | 图像溢出边界采样（自动 pad） |
| `ndarray_over_scan_wrapper.py` | 多维数组溢出边界读写 |
| `multi_scale_patch_result.py` | 多尺度图块结果：滑窗合并、轮廓提取 |
| `multi_scale_large_image_result.py` | 大图扫描结果类：与 `MultiScalePatchResult` 组合使用 |
| `image_free_gauss_fusion_wrapper.py` | 大图高斯融合：分区加速、硬盘读取 |
| `im_coords_affine_enhance_tool.py` | 图像 + 坐标联合仿射增强 |

### 🗂️ 数据读写

| 模块 | 说明 |
|------|------|
| `json_tool.py` | JSON 读写（保留中文） |
| `csv_tool.py` | CSV 读写 |
| `xlsx_tool.py` | Excel 读写（openpyxl） |
| `awkward_tool.py` | 从 CSV / Pandas / Excel 加载为 awkward 数组 |
| `io_utils.py` | 二进制文件读写：类型、字符串、向量（与 C++ 版配对） |
| `download_tool.py` | 断点续传下载（SHA1 校验） |

### 🏷️ 标注工具

| 模块 | 说明 |
|------|------|
| `imagescope_xml_utils.py` | Aperio ImageScope XML 读写（轮廓、方框、箭头、椭圆） |
| `asap_xml_utils.py` | ASAP XML 读写（轮廓、方框、点、样条、点集） |
| `qupath_label_tool.py` | QuPath GeoJSON 读写（轮廓、多点、多边形） |

### 📦 数据集

| 模块 | 说明 |
|------|------|
| `dataset/dataset.py` | 虚拟基类 |
| `dataset/coco_dataset.py` | COCO 数据集（边界框、像素分割、关键点） |
| `dataset/voc_dataset.py` | VOC 数据集（边界框） |
| `dataset/_cocostuffhelper.py` | COCO Stuff 分割格式转换辅助函数 |

### 🧰 其他工具

| 模块 | 说明 |
|------|------|
| `list_tool.py` | 列表操作：批量取/设/删/弹出、分割、分组、填充 |
| `numpy_tool.py` | NumPy 工具：one-hot、最短配对、figure 转数组 |
| `path_tool.py` | 路径操作：拆分、拼接、后缀替换、递归查找 |
| `str_half2full_tool.py` | 全角半角互转（中日韩字符） |
| `plot_tool.py` | Matplotlib 绘图：散点图、自定义颜色 |
| `tissue_tool.py` | 病理图组织区域轮廓提取 |
| `preload_generator.py` | 预加载缓存生成器（多线程） |
| `universal_batch_generator.py` | 多输出生成器批量打包（线程安全） |
| `utils.py` | 杂项：值域映射、TensorFlow Session、临时 sys.path |
| `void_cls.py` | 空类 `VC`，用作占位符 |
| `simple_bucket.py` | 简单数据桶：索引 + 数据分离存储 |

---

## 安装

```bash
pip install -e .
```

无需额外依赖（`setup.py` 中 `install_requires` 为空）。各模块按需 import，使用前请确保已安装对应库（如 `opencv-python`、`numpy`、`shapely`、`openslide` 等）。
