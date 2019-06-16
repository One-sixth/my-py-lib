# my_py_lib
Just a tiny lib to help I use python.  
If it helps you, it would be great.  
  
The code estimates that there are many bug and there are a lot of irregularities.  
so please understand and check the code before using it.  
I only tested it on the windows platform.  


# What is inside
## coord_tool.py
Currently there are only a variety of coordinate format transformations, such as xywh_to_x1y1x2y2.  

## im_tool.py
For example, draw a bounding box onto the image, a simple image pad.  

## utils.py
Miscellaneous, rarely used.  

## dataset
### coco_dataset.py
### voc_dataset.py
### dataset.py
This is virtual base class, other dataset need to inherit it.  
Currently voc dataset and coco dataset are implemented.  
Where voc dataset can work on most datasets similar to voc 2012 format (uncertain)  
Coco estimates that it can only work on Windows platforms, recompiling pycocotools can work on Linux.  
The voc dataset implements the bounding box output.  
The coco dataset implements bounding boxes, pixel segmentation, and human bone feature point output.  