import os
import random
import numpy as np

import cv2
from ultralytics import YOLO

from tracker import Tracker

mot_path = os.path.join('.', 'data', 'MOT17')
