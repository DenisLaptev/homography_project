from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
# GOOD_MATCH_PERCENT = 0.65
# GOOD_MATCH_PERCENT = 0.25

FRAMES_DELTA = 2

# Define the codec and create VideoWriter object

# width_initial=1280, height_initial=720
# width_resized=640, height_resized=360


# width=640, height=360
out_maximum_Edges = cv2.VideoWriter('maximum_SobelEdges_delta=2.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10,
                              (640, 360))

# width=640, height=360
out_difference_int_abs = cv2.VideoWriter('difference_int_abs_SobelEdges_delta=2.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10,
                              (640, 360))

# # width=640, height=360
# out_maximum_Edges = cv2.VideoWriter('maximum_CannyEdges_delta=1.avi',
#                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
#                               10,
#                               (640, 360))
#
# # width=640, height=360
# out_difference_int_abs = cv2.VideoWriter('difference_int_abs_CannyEdges_delta=1.avi',
#                               cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
#                               10,
#                               (640, 360))



def get_roi(image, title):
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # return image_gray
    return image


def alignImages(im, imReference, im1, im2):
    # Convert images to grayscale
    # im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    # im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    im1Gray = im1
    im2Gray = im2

    # Detect ORB features and compute descriptors.
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)

    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:numGoodMatches]

    # Draw top matches
    imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

    # Use homography
    # height, width, channels = imReference.shape
    height, width = imReference.shape
    im1Reg = cv2.warpPerspective(im, h, (width, height))

    return im1Reg, h, imMatches
    # return imMatches


def get_difference(frame_initial, frame_aligned):
    frame_initial_float = frame_initial.astype(float)
    frame_aligned_float = frame_aligned.astype(float)
    print('frame_initial_float=', frame_initial_float)
    print('frame_aligned_float=', frame_aligned_float)

    frames_difference_float = frame_initial_float - frame_aligned_float
    print('frames_difference_float=', frames_difference_float)

    frames_difference_float_abs = np.absolute(frames_difference_float)
    print('frames_difference_float_abs=', frames_difference_float_abs)

    frames_difference_int_abs = frames_difference_float_abs.astype(int)
    print('frames_difference_int_abs=', frames_difference_int_abs)

    return frames_difference_int_abs.astype('uint8')


cap1 = cv2.VideoCapture('../VL_left1.avi')
cap2 = cv2.VideoCapture('../VL_left1.avi')

frame1_count = 0
frame2_count = 0

while (cap1.isOpened()):
    ret1, frame1 = cap1.read()

    if ret1 == True:

        height1, width1, _ = frame1.shape
        dim1 = (width1 // 2, height1 // 2)
        frame1 = cv2.resize(frame1, dim1)

        if frame1_count >= FRAMES_DELTA:
            ret2, frame2 = cap2.read()

            print('frame1_count=', frame1_count)
            print('frame2_count=', frame2_count)
            print('frame1_count-frame2_count=', frame1_count - frame2_count)

            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # height2, width2, _ = frame2.shape
            height2, width2 = frame2.shape
            dim2 = (width2 // 2, height2 // 2)
            frame2 = cv2.resize(frame2, dim2)

            imReference = frame1
            im = frame2

            imReference_roi = get_roi(image=imReference, title='ref_image')
            im_roi = get_roi(image=im, title='work_image')

            # imReg, h = alignImages(im, imReference)
            imReg, h, imMatches = alignImages(im, imReference, im_roi, imReference_roi)
            # imMatches = alignImages(im, imReference, im_roi, imReference_roi)

            frame2_alligned = imReg
            # Print estimated homography
            print("Estimated homography : \n", h)

            # cv2.imshow('imMatches', imMatches)
            cv2.imshow('frame2', frame2)
            cv2.imshow('frame2_alligned', frame2_alligned)

            #Sobel
            frame2_Edges = cv2.Sobel(frame2, cv2.CV_8U, 1, 0, ksize=5)
            frame2_alligned_Edges  = cv2.Sobel(frame2_alligned, cv2.CV_8U, 1, 0, ksize=5)

            # # Canny
            # frame2_Edges = cv2.Canny(frame2, 100, 200)
            # frame2_alligned_Edges = cv2.Canny(frame2_alligned, 100, 200)

            cv2.imshow('frame2_Edges', frame2_Edges)
            cv2.imshow('frame2_alligned_Edges', frame2_alligned_Edges)

            frame2_difference_int_abs = get_difference(frame_initial=frame2_Edges,
                                                       frame_aligned=frame2_alligned_Edges)

            frame2_maximum_Edges = np.maximum(frame2_Edges,
                                              frame2_alligned_Edges)

            cv2.imshow('frame2_maximum_Edges', frame2_maximum_Edges)
            cv2.imshow('frame2_difference_int_abs', frame2_difference_int_abs.astype('uint8'))

            # write 2 output videos
            # out.write(imMatches)
            # out_maximum_Edges.write(frame2_maximum_Edges)
            # out_difference_int_abs.write(frame2_difference_int_abs)

            # save images for Frame_1500
            if frame1_count == 1500:
                # cv2.imwrite('maximum_SobelEdges_for_Frame_1500_delta=1.jpg', frame2_maximum_Edges)
                # cv2.imwrite('difference_int_abs_SobelEdges_Frame_1500_delta=1.jpg', frame2_difference_int_abs)
                # break

                cv2.imwrite('maximum_SobelEdges_for_Frame_1500_delta=2.jpg', frame2_maximum_Edges)
                cv2.imwrite('difference_int_abs_SobelEdges_Frame_1500_delta=2.jpg', frame2_difference_int_abs)
                break
                # cv2.imwrite('maximum_CannyEdges_for_Frame_1500_delta=1.jpg', frame2_maximum_Edges)
                # cv2.imwrite('difference_int_abs_CannyEdges_Frame_1500_delta=1.jpg', frame2_difference_int_abs)
                #
                # cv2.imwrite('maximum_CannyEdges_for_Frame_1500_delta=2.jpg', frame2_maximum_Edges)
                # cv2.imwrite('difference_int_abs_CannyEdges_Frame_1500_delta=2.jpg', frame2_difference_int_abs)

            frame2_count = frame2_count + 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        frame1_count = frame1_count + 1

    else:
        break

print('frame1_count=', frame1_count)
print('frame2_count=', frame2_count)

cap1.release()
cap2.release()
out_maximum_Edges.release()
out_difference_int_abs.release()
cv2.destroyAllWindows()
