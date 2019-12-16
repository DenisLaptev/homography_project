from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15
#GOOD_MATCH_PERCENT = 0.35


# GOOD_MATCH_PERCENT = 0.25


def get_roi(image, title):

    height, width, channels = image.shape

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow(title, image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    mask = np.zeros_like(image_gray)
    mask[height // 3:2 * height // 3, width // 3:2 * width // 3] = 1
    print(mask)
    cv2.imshow(title, mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    masked_roi = image_gray*mask
    cv2.imshow('masked_roi', masked_roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return masked_roi


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
    # cv2.imwrite("matches_286.jpg", imMatches)
    # cv2.imwrite("matches_433.jpg", imMatches)
    # cv2.imwrite("matches_1353.jpg", imMatches)
    cv2.imwrite("matches_3061.jpg", imMatches)

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

    return im1Reg, h


if __name__ == '__main__':
    # Read reference image
    # refFilename = "../images/_VL_Nagaraju_frame_287.jpg"
    # refFilename = "../images/_VL_Nagaraju_frame_434.jpg"
    # refFilename = "../images/_VL_satishreddyb_frame_1354.jpg"
    refFilename = "../images/_VL_tsantosh_frame_3062.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    # imFilename = "../images/_VL_Nagaraju_frame_286.jpg"
    # imFilename = "../images/_VL_Nagaraju_frame_433.jpg"
    # imFilename = "../images/_VL_satishreddyb_frame_1353.jpg"
    imFilename = "../images/_VL_tsantosh_frame_3061.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.

    im_roi = get_roi(image=im, title='work_image')
    imReference_roi = get_roi(image=imReference, title='ref_image')

    # imReg, h = alignImages(im, imReference)
    imReg, h = alignImages(im, imReference, im_roi, imReference_roi)

    # Write aligned image to disk.
    # outFilename = "aligned_286.jpg"
    # outFilename = "aligned_433.jpg"
    # outFilename = "aligned_1353.jpg"
    outFilename = "aligned_3061.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)
