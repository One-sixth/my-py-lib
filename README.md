# my_py_lib
Just a tiny lib to help I use python.<br>
If it helps you, it would be great.<br>
<br>
The code estimates that there are many bug and there are a lot of irregularities.<br>
so please understand and check the code before using it.<br>
I only tested it on the windows platform.<br>


# What is inside
## coord_tool.py
Currently there are only a variety of coordinate format transformations, such as xywh_to_x1y1x2y2.

## im_tool.py
For example, draw a bounding box onto the image, a simple image pad.

## utils.py
Miscellaneous, rarely used.

## pycocotools
For coco dataset. I am compiling on windows using vs2017, which may not work on linux.<br>
This I copy from https://github.com/cocodataset/cocoapi<br>

## dataset
### coco_dataset.py
### voc_dataset.py
### dataset.py      This is virtual base class, other dataset need to inherit it.
Currently voc dataset and coco dataset are implemented.<br>
Where voc dataset can work on most datasets similar to voc 2012 format (uncertain)<br>
Coco estimates that it can only work on Windows platforms, recompiling pycocotools can work on Linux.<br>
The voc dataset implements the bounding box output.<br>
The coco dataset implements bounding boxes, pixel segmentation, and human bone feature point output.<br>