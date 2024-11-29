!nvidia-smi 

!pip install ultralytics

import ultralytics
ultralytics.checks()

from ultralytics import YOLO
from IPython.display import display, Image  

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="y8WlkFc8TYCHsK3YgF4X")
project = rf.workspace("nasiruddin-thander-3daow").project("new_rgb_plate")
version = project.version(1)
dataset = version.download("yolov11")
                

dataset.location

!yolo task=detect mode=train data={dataset.location}/data.yaml model="yolo11n.pt" epochs=25 imgsz=640

from IPython.display import Image
Image('/content/runs/detect/train2/confusion_matrix.png', width=600)

from IPython.display import Image
Image('/content/runs/detect/train2/labels.jpg', width=600)

from IPython.display import Image
Image('/content/runs/detect/train2/train_batch1.jpg', width=600)

from IPython.display import Image
Image('/content/runs/detect/train2/train_batch2.jpg', width=600)

!yolo task=detect mode=val model="/content/runs/detect/train2/weights/best.pt" data={dataset.location}/data.yaml 

!yolo task=detect mode=predict model="/content/runs/detect/train2/weights/best.pt" conf=0.25 source={dataset.location}/test/images save=true

import glob
import os
from IPython.display import Image as IPyImage, display


latest_folder = max(glob.glob('/content/runs/detect/predict*'), key=os.path.getmtime)
for img_path in glob.glob(f'{latest_folder}/*.jpg')[1:4]:
  display(IPyImage(filename=img_path,width=600))
print("\n")
