import cv2
import numpy as np
# full_pb_path = "./results/models/forward_graph.pb"
# opencv_net = cv2.dnn.readNetFromTensorflow(full_pb_path)


full_onnx_path = "./results/models/forward_graph.pb"
opencv_net = cv2.dnn.readNetFromTensorflow(full_onnx_path, full_onnx_path + "txt")

