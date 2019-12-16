from __future__ import print_function
import cv2
import numpy as np

cap = cv2.VideoCapture('../VL_left1.avi')

frame_count = 0
while (cap.isOpened()):

    ret, frame = cap.read()

    if ret == True:

        #save dataset for NN
        if frame_count > 1330 and frame_count < 1830:
            title = "../dataset_for_NN/Frame_" + str(frame_count) + ".png"
            cv2.imwrite(title, frame)

        if frame_count == 2000:
            break

        frame_count = frame_count + 1
        print('frame_count=', frame_count)

    else:
        break

cap.release()
cv2.destroyAllWindows()
