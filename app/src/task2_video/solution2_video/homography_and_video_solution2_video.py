from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
# GOOD_MATCH_PERCENT = 0.65
# GOOD_MATCH_PERCENT = 0.25

FRAMES_DELTA = 1

# Define the codec and create VideoWriter object

# width_initial=1280, height_initial=720
# width_resized=640, height_resized=360

# width=640*2, height=360
out_matches = cv2.VideoWriter('matches.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10,
                              (1280, 360))
# width=640, height=360
out_aligned = cv2.VideoWriter('aligned_images.avi',
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                              10,
                              (640, 360))


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
    height, width, channels = imReference.shape
    im1Reg = cv2.warpPerspective(im, h, (width, height))

    return im1Reg, h, imMatches
    # return imMatches


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
            height2, width2, _ = frame2.shape
            dim2 = (width2 // 2, height2 // 2)
            frame2 = cv2.resize(frame2, dim2)

            # print('frame1_count=',frame1_count)
            # print('frame2_count=',frame2_count)
            print('frame1_count-frame2_count=', frame1_count - frame2_count)

            imReference = frame1
            im = frame2

            imReference_roi = get_roi(image=imReference, title='ref_image')
            im_roi = get_roi(image=im, title='work_image')

            imReg, h, imMatches = alignImages(im, imReference, im_roi, imReference_roi)

            # Print estimated homography
            print("Estimated homography : \n", h)

            # write matches and the flipped frame
            out_matches.write(imMatches)
            out_aligned.write(imReg)

            # show matches and the flipped frame
            cv2.imshow('imMatches', imMatches)
            cv2.imshow('imReg', imReg)

            # save images for Frame_1500
            if frame1_count == 1500:
                cv2.imwrite('matching_for_Frame_1500.jpg', imMatches)
                cv2.imwrite('aligned_Frame_1500.jpg', imReg)

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
out_matches.release()
out_aligned.release()
cv2.destroyAllWindows()
