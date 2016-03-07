This module contains the class vpCascadeClassifier which permits to:
- load a classifier data, detect the objects in the image
- match the detections over time and assign them an id in such a way that an same id at t and t+1 should correspond to the same object
- track a lost object using the ViSP SSD template tracker
