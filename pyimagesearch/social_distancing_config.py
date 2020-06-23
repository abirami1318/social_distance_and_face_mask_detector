# base path to YOLO directory
MODEL_PATH = "yolo-coco"

# initialize minimum probability to filter weak detections along with
# the threshold when applying non-maxima suppression
MIN_CONF = 0.3
NMS_THRESH = 0.3

# boolean indicating if NVIDIA CUDA GPU should be used
USE_GPU = False

# define the minimum safe distance (in pixels) that two people can be
# from each other
MIN_DISTANCE0 = 60  #if both the people are not wearing mask
MIN_DISTANCE2 = 30  #if both the people are wearing mask
MIN_DISTANCE1 = 40  # if one person is wearing mask and other is not wearing mask or the other person face is not detected
MIN_DISTANCE = 50  #if both the people faces are not detected