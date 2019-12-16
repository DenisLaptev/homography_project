import numpy as np
import cv2

amean = [0.485, 0.456, 0.406]
astd = [0.229, 0.224, 0.225]

path_to_image_from_NN = '../from_nn/images/Frame_1331.png'
path_to_map1_from_NN = '../from_nn/images/Frame_1331_1_avg.png'
path_to_map2_from_NN = '../from_nn/images/Frame_1331_2_avg.png'
path_to_map3_from_NN = '../from_nn/images/Frame_1331_3_avg.png'
path_to_map4_from_NN = '../from_nn/images/Frame_1331_4_avg.png'
path_to_image_for_NN = '../dataset_for_NN/Frame_1331.png'

image_from_NN = cv2.imread(path_to_image_from_NN)
map1_from_NN = cv2.imread(path_to_map1_from_NN)
map2_from_NN = cv2.imread(path_to_map2_from_NN)
map3_from_NN = cv2.imread(path_to_map3_from_NN)
map4_from_NN = cv2.imread(path_to_map4_from_NN)
image_for_NN = cv2.imread(path_to_image_for_NN)

print('image_from_NN.shape=', image_from_NN.shape)
print('map1_from_NN.shape=', map1_from_NN.shape)
print('map2_from_NN.shape=', map2_from_NN.shape)
print('map3_from_NN.shape=', map3_from_NN.shape)
print('map4_from_NN.shape=', map4_from_NN.shape)
print('image_for_NN.shape=', image_for_NN.shape)

height, width, _ = image_for_NN.shape

image_from_NN = cv2.resize(image_from_NN, (width, height))
map1_from_NN = cv2.resize(map1_from_NN, (width, height))
map2_from_NN = cv2.resize(map2_from_NN, (width, height))
map3_from_NN = cv2.resize(map3_from_NN, (width, height))
map4_from_NN = cv2.resize(map4_from_NN, (width, height))

ret1, map1_thresh = cv2.threshold(map1_from_NN, 100, 255, cv2.THRESH_BINARY_INV)
ret2, map2_thresh = cv2.threshold(map2_from_NN, 100, 255, cv2.THRESH_BINARY_INV)
ret3, map3_thresh = cv2.threshold(map3_from_NN, 100, 255, cv2.THRESH_BINARY)
ret4, map4_thresh = cv2.threshold(map4_from_NN, 100, 255, cv2.THRESH_BINARY)

# result = map1_from_NN+map2_from_NN+map3_from_NN+map4_from_NN

# make blue
map1_from_NN[:, :, 1] = 0
map1_from_NN[:, :, 2] = 0

# make green
map2_from_NN[:, :, 0] = 0
map2_from_NN[:, :, 2] = 0

# make red
map3_from_NN[:, :, 0] = 0
map3_from_NN[:, :, 1] = 0

# make yellow
map4_from_NN[:, :, 0] = 0

print("--------after resize----------")

print('image_from_NN.shape=', image_from_NN.shape)
print('map1_from_NN.shape=', map1_from_NN.shape)
print('map2_from_NN.shape=', map2_from_NN.shape)
print('map3_from_NN.shape=', map3_from_NN.shape)
print('map4_from_NN.shape=', map4_from_NN.shape)
print('image_for_NN.shape=', image_for_NN.shape)

# cv2.imshow('image_from_NN', image_from_NN)
# cv2.imshow('map1_from_NN', map1_from_NN)
# cv2.imshow('map2_from_NN', map2_from_NN)
# cv2.imshow('map3_from_NN', map3_from_NN)
# cv2.imshow('map4_from_NN', map4_from_NN)
# cv2.imshow('image_for_NN', image_for_NN)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('map1_thresh', map1_thresh)
# cv2.imshow('map2_thresh', map2_thresh)
# cv2.imshow('map3_thresh', map3_thresh)
# cv2.imshow('map4_thresh', map4_thresh)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

result = map1_from_NN + map2_from_NN + map3_from_NN + map4_from_NN
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()

result = cv2.add(image_for_NN, result)
# result = cv2.addWeighted(image_for_NN, 0.7, result, 0.3, 0)
# result = result+image_for_NN
cv2.imshow('result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
