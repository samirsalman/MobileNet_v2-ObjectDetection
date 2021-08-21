# MobileNet v2 vs YOLO v3 ** Object Detection Task

In this project, I build a ready-to-use solution to experiment with Object Detection models. You can use the most famous object detection models with live webcam videos or with YouTube videos.

Supported models (updating):
- YOLO v3
- MobileNet v2

| | |
|:-------------------------:|:-------------------------:|
|YOLO v3 ![YOLOv3](https://github.com/samirsalman/Yolo_v3-VS-MobileNet_v2-ObjectDetection/blob/main/tests/yolo.gif) | MOBILENET v2  ![mobilenet](https://github.com/samirsalman/Yolo_v3-VS-MobileNet_v2-ObjectDetection/blob/main/tests/mobilenet.gif)|

## Project Structure

- models -> This directory contains all supported models (weights and configurations)
  - Models.py -> contains all models path to allow the imports

- utils -> Contains the utils files and function used in the project
  - YoutubeDownloader.py -> Allow the user to download videos from YouTube

- DetectionModel.py -> Class that allows to init a Detection Model

- main.py -> main file, where the user can specify the model to use and the test mode (camera or YT video)


## Models

## Author

Samir Salman
