from __future__ import print_function
import cv2
import numpy as np

MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


# GOOD_MATCH_PERCENT = 0.65
# GOOD_MATCH_PERCENT = 0.25


def get_roi(image, title):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow(title, image_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image_gray


def alignImages(im, imReference,
                im1, im2,
                map11, map21, map31, map41,
                map12, map22, map32, map42,
                map12name, map22name, map32name, map42name):
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
    cv2.imwrite("matches_286.jpg", imMatches)
    # cv2.imwrite("matches_433.jpg", imMatches)
    # cv2.imwrite("matches_1353.jpg", imMatches)
    # cv2.imwrite("matches_3061.jpg", imMatches)

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

    print('map11=', map11)
    print('map12=', map12)

    # Use homography for maps
    map1_avg = calc_Avg_map(mapX1=map11, mapX2=map12, homography=h, mapX2name=map12name)
    map2_avg = calc_Avg_map(mapX1=map21, mapX2=map22, homography=h, mapX2name=map22name)
    map3_avg = calc_Avg_map(mapX1=map31, mapX2=map32, homography=h, mapX2name=map32name)
    map4_avg = calc_Avg_map(mapX1=map41, mapX2=map42, homography=h, mapX2name=map42name)

    return im1Reg, h, map1_avg, map2_avg, map3_avg, map4_avg


def calc_Avg_map(mapX1, mapX2, homography, mapX2name):
    # Use homography for maps
    h_m, w_m, ch_m = mapX2.shape
    h = homography
    mapX1Reg = cv2.warpPerspective(mapX1, h, (w_m, h_m))
    mapX_avg = (mapX1Reg // 2 + mapX2 // 2)

    cv2.imshow('mapX_avg', mapX_avg)
    cv2.imwrite("../Avg/" + mapX2name, mapX_avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return mapX_avg


if __name__ == '__main__':
    # Read reference image
    refFilename = "../images/_VL_Nagaraju_frame_287.jpg"
    # refFilename = "../images/_VL_Nagaraju_frame_434.jpg"
    # refFilename = "../images/_VL_satishreddyb_frame_1354.jpg"
    # refFilename = "../images/_VL_tsantosh_frame_3062.jpg"
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)

    # Read image to be aligned
    imFilename = "../images/_VL_Nagaraju_frame_286.jpg"
    # imFilename = "../images/_VL_Nagaraju_frame_433.jpg"
    # imFilename = "../images/_VL_satishreddyb_frame_1353.jpg"
    # imFilename = "../images/_VL_tsantosh_frame_3061.jpg"
    print("Reading image to align : ", imFilename)
    im = cv2.imread(imFilename, cv2.IMREAD_COLOR)

    # 286
    map11name = "_VL_Nagaraju_frame_286_1_avg.png"
    map21name = "_VL_Nagaraju_frame_286_2_avg.png"
    map31name = "_VL_Nagaraju_frame_286_3_avg.png"
    map41name = "_VL_Nagaraju_frame_286_4_avg.png"
    map12name = "_VL_Nagaraju_frame_287_1_avg.png"
    map22name = "_VL_Nagaraju_frame_287_2_avg.png"
    map32name = "_VL_Nagaraju_frame_287_3_avg.png"
    map42name = "_VL_Nagaraju_frame_287_4_avg.png"

    # # 433
    # map11name = "_VL_Nagaraju_frame_433_1_avg.png"
    # map21name = "_VL_Nagaraju_frame_433_2_avg.png"
    # map31name = "_VL_Nagaraju_frame_433_3_avg.png"
    # map41name = "_VL_Nagaraju_frame_433_4_avg.png"
    # map12name = "_VL_Nagaraju_frame_434_1_avg.png"
    # map22name = "_VL_Nagaraju_frame_434_2_avg.png"
    # map32name = "_VL_Nagaraju_frame_434_3_avg.png"
    # map42name = "_VL_Nagaraju_frame_434_4_avg.png"

    # # 1353
    # map11name = "_VL_satishreddyb_frame_1353_1_avg.png"
    # map21name = "_VL_satishreddyb_frame_1353_2_avg.png"
    # map31name = "_VL_satishreddyb_frame_1353_3_avg.png"
    # map41name = "_VL_satishreddyb_frame_1353_4_avg.png"
    # map12name = "_VL_satishreddyb_frame_1354_1_avg.png"
    # map22name = "_VL_satishreddyb_frame_1354_2_avg.png"
    # map32name = "_VL_satishreddyb_frame_1354_3_avg.png"
    # map42name = "_VL_satishreddyb_frame_1354_4_avg.png"

    # # 3061
    # map11name = "_VL_tsantosh_frame_3061_1_avg.png"
    # map21name = "_VL_tsantosh_frame_3061_2_avg.png"
    # map31name = "_VL_tsantosh_frame_3061_3_avg.png"
    # map41name = "_VL_tsantosh_frame_3061_4_avg.png"
    # map12name = "_VL_tsantosh_frame_3062_1_avg.png"
    # map22name = "_VL_tsantosh_frame_3062_2_avg.png"
    # map32name = "_VL_tsantosh_frame_3062_3_avg.png"
    # map42name = "_VL_tsantosh_frame_3062_4_avg.png"

    map11 = cv2.imread("../images/" + map11name, cv2.IMREAD_COLOR)
    map21 = cv2.imread("../images/" + map21name, cv2.IMREAD_COLOR)
    map31 = cv2.imread("../images/" + map31name, cv2.IMREAD_COLOR)
    map41 = cv2.imread("../images/" + map41name, cv2.IMREAD_COLOR)
    map12 = cv2.imread("../images/" + map12name, cv2.IMREAD_COLOR)
    map22 = cv2.imread("../images/" + map22name, cv2.IMREAD_COLOR)
    map32 = cv2.imread("../images/" + map32name, cv2.IMREAD_COLOR)
    map42 = cv2.imread("../images/" + map42name, cv2.IMREAD_COLOR)

    print("Aligning images ...")
    # Registered image will be resotred in imReg.
    # The estimated homography will be stored in h.

    im_roi = get_roi(image=im, title='work_image')
    imReference_roi = get_roi(image=imReference, title='ref_image')

    print('imReference.shape=', imReference.shape)
    print('im.shape=', im.shape)
    print('imReference_roi.shape=', imReference_roi.shape)
    print('im_roi.shape=', im_roi.shape)

    print('map11.shape=', map11.shape)
    print('map12.shape=', map12.shape)

    print('map21.shape=', map21.shape)
    print('map22.shape=', map22.shape)

    print('map31.shape=', map31.shape)
    print('map32.shape=', map32.shape)

    print('map41.shape=', map41.shape)
    print('map42.shape=', map42.shape)

    # imReg, h = alignImages(im, imReference)
    imReg, h, map1_avg, map2_avg, map3_avg, map4_avg = alignImages(im, imReference,
                                                                   im_roi, imReference_roi,
                                                                   map11, map21, map31, map41,
                                                                   map12, map22, map32, map42,
                                                                   map12name, map22name, map32name, map42name)

    # Write aligned image to disk.
    outFilename = "aligned_286.jpg"
    # outFilename = "aligned_433.jpg"
    # outFilename = "aligned_1353.jpg"
    # outFilename = "aligned_3061.jpg"
    print("Saving aligned image : ", outFilename)
    cv2.imwrite(outFilename, imReg)

    # Print estimated homography
    print("Estimated homography : \n", h)

    print('---------------Avg_maps-------------------')
    print('map1_avg=', map1_avg)
    print('map2_avg=', map2_avg)
    print('map3_avg=', map3_avg)
    print('map4_avg=', map4_avg)
