# :seedling: Testing branch (laptop's lab)

This branch aims at testing scripts available in the default one.
Layout:
| PC   |  Nvidia CUDA |
|----------|:-------------:|
| Asus | :white_check_mark:  |  


# :rocket: Roadmap

- [x] create pkg for custom messages
- [x] upload py scripts aimed at testing purposes 
- [x] Create a ros node that:
    - [x] Reads the image 
    - [x] Performs the detection
    - [x] publish commands 
  
  
# repo's hierarchy -- DETR

```sh
.
├── D-Drone_v2
│   ├── dataset
│   │   └── test
│   │       └── test
│   │           ├── Image0012.jpg
│   │           ├── Image0015.jpg
│   │           ├── Image0028.jpg
│   │           ├── Image0029.jpg
│   │           ├── Image0031.jpg
.   .           .   .............
.   .           .   .............
.   .           .   .............
│   │           ├── Image1986.jpg
│   │           ├── Image1990.jpg
│   │           ├── Image1991.jpg
│   │           ├── Image1997.jpg
│   │           └── test_image.jpg
│   ├── datasets
│   │   ├── coco_eval.py
│   │   ├── coco_panoptic.py
│   │   ├── coco.py
│   │   ├── drone.py
│   │   ├── __init__.py
│   │   ├── panoptic_eval.py
│   │   ├── __pycache__
│   │   │   ├── coco.cpython-37.pyc
│   │   │   ├── coco.cpython-38.pyc
│   │   │   ├── coco_eval.cpython-37.pyc
│   │   │   ├── drone.cpython-37.pyc
│   │   │   ├── drone.cpython-38.pyc
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── panoptic_eval.cpython-37.pyc
│   │   │   ├── transforms.cpython-37.pyc
│   │   │   └── transforms.cpython-38.pyc
│   │   └── transforms.py
│   ├── Detect
│   │   ├── expImage0012.jpg
│   │   ├── expImage0015.jpg
│   │   ├── expImage0028.jpg
│   │   ├── expImage0029.jpg
│   │   ├── expImage0031.jpg
│   │   ├── expImage0036.jpg
│   │   ├── expImage0038.jpg
│   │   ├── expImage0052.jpg
.   .   .           .   ....
.   .   .           .   ....
.   .   .           .   ....
│   │   ├── expImage1954.jpg
│   │   ├── expImage1958.jpg
│   │   ├── expImage1963.jpg
│   │   ├── expImage1974.jpg
│   │   ├── expImage1983.jpg
│   │   ├── expImage1986.jpg
│   │   ├── expImage1990.jpg
│   │   ├── expImage1991.jpg
│   │   ├── expImage1997.jpg
│   │   └── exptest_image.jpg
│   ├── DETR
│   │   ├── DEMO.ipynb
│   │   ├── DeTr_v2.ipynb
│   │   └── Test_DETR.ipynb
│   ├── LICENSE
│   ├── models
│   │   ├── backbone.py
│   │   ├── detr.py
│   │   ├── __init__.py
│   │   ├── matcher.py
│   │   ├── position_encoding.py
│   │   ├── __pycache__
│   │   │   ├── backbone.cpython-37.pyc
│   │   │   ├── backbone.cpython-38.pyc
│   │   │   ├── detr.cpython-37.pyc
│   │   │   ├── detr.cpython-38.pyc
│   │   │   ├── __init__.cpython-37.pyc
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── matcher.cpython-37.pyc
│   │   │   ├── matcher.cpython-38.pyc
│   │   │   ├── position_encoding.cpython-37.pyc
│   │   │   ├── position_encoding.cpython-38.pyc
│   │   │   ├── segmentation.cpython-37.pyc
│   │   │   ├── segmentation.cpython-38.pyc
│   │   │   ├── transformer.cpython-37.pyc
│   │   │   └── transformer.cpython-38.pyc
│   │   ├── segmentation.py
│   │   └── transformer.py
│   ├── Output
│   │   ├── 1.png
│   │   ├── 2.png
│   │   ├── 3.png
│   │   ├── 4.png
│   │   ├── 5.png
│   │   ├── 6.png
│   │   ├── Drone_detr.png
│   │   ├── Drone_yolov4.png
│   │   ├── Drone_yolov5.png
│   │   └── metrics.PNG
│   ├── output_labels
│   │   ├── Image0012.txt
│   │   ├── Image0015.txt
│   │   ├── Image0028.txt
│   │   ├── Image0029.txt
│   │   ├── Image0031.txt
│   │   ├── Image0036.txt
│   │   ├── Image0063.txt
│   │   ├── Image0090.txt
│   │   ├── Image0099.txt
│   │   ├── Image0107.txt
.   .   .   .............
.   .   .   .............
.   .   .   .............
.   .   .   .............
│   │   ├── Image1986.txt
│   │   ├── Image1990.txt
│   │   ├── Image1991.txt
│   │   └── Image1997.txt
│   ├── output_model
│   │   └── output_model
│   │       ├── checkpoint.pth
│   │       ├── eval
│   │       │   ├── 000.pth
│   │       │   ├── 050.pth
│   │       │   └── latest.pth
│   │       └── log.txt
│   ├── README.md
│   ├── requirements.txt
│   ├── test.py
│   ├── util
│   │   ├── box_ops.py
│   │   ├── __init__.py
│   │   ├── misc.py
│   │   ├── plot_utils.py
│   │   └── __pycache__
│   │       ├── box_ops.cpython-37.pyc
│   │       ├── box_ops.cpython-38.pyc
│   │       ├── __init__.cpython-37.pyc
│   │       ├── __init__.cpython-38.pyc
│   │       ├── misc.cpython-37.pyc
│   │       └── misc.cpython-38.pyc
│   ├── YOLOv4
│   │   ├── Demo.ipynb
│   │   ├── Test.ipynb
│   │   └── Training.ipynb
│   └── YOLOv5
│       ├── DEMO.ipynb
│       └── Training.ipynb
└── repo_hierarchy.txt

19 directories, 678 files

```
