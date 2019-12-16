import numpy as np
import cv2

# cap = cv2.VideoCapture(0)

# Get the Default resolutions
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))


MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15

# Define the codec and filename.
out = cv2.VideoWriter('video_with_AVERAGED_colored_lanes.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))


def alignImages(im1, im2):
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

    return h, imMatches
    # return imMatches

# def calc_Avg_map(mapX1, mapX2, homography, mapX2name):
def calc_Avg_map(mapX1, mapX2, homography):
    # Use homography for maps
    h_m, w_m, ch_m = mapX2.shape
    h = homography
    mapX1Reg = cv2.warpPerspective(mapX1, h, (w_m, h_m))
    mapX_avg = (mapX1Reg.astype(float)/2 + mapX2.astype(float)/2)
    mapX_avg = mapX_avg.astype('uint8')

    # cv2.imshow('mapX_avg', mapX_avg)
    # cv2.imwrite("../Avg/" + mapX2name, mapX_avg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mapX_avg


def get_frame_with_lanes(path_to_map1_from_NN, path_to_map2_from_NN,
                         path_to_map3_from_NN, path_to_map4_from_NN,
                         path_to_image_for_NN,
                         path_to_map1_from_NN_previous,
                         path_to_map2_from_NN_previous,
                         path_to_map3_from_NN_previous,
                         path_to_map4_from_NN_previous,
                         path_to_image_for_NN_previous):
    map1_from_NN = cv2.imread(path_to_map1_from_NN)
    map2_from_NN = cv2.imread(path_to_map2_from_NN)
    map3_from_NN = cv2.imread(path_to_map3_from_NN)
    map4_from_NN = cv2.imread(path_to_map4_from_NN)

    map1_from_NN_previous = cv2.imread(path_to_map1_from_NN_previous)
    map2_from_NN_previous = cv2.imread(path_to_map2_from_NN_previous)
    map3_from_NN_previous = cv2.imread(path_to_map3_from_NN_previous)
    map4_from_NN_previous = cv2.imread(path_to_map4_from_NN_previous)

    image_for_NN = cv2.imread(path_to_image_for_NN)
    image_for_NN_previous = cv2.imread(path_to_image_for_NN_previous)

    print('map1_from_NN.shape=', map1_from_NN.shape)
    print('map2_from_NN.shape=', map2_from_NN.shape)
    print('map3_from_NN.shape=', map3_from_NN.shape)
    print('map4_from_NN.shape=', map4_from_NN.shape)
    print('image_for_NN.shape=', image_for_NN.shape)
    print('image_for_NN.shape=', image_for_NN.shape)

    height, width, _ = image_for_NN.shape

    map1_from_NN = cv2.resize(map1_from_NN, (width, height))
    map2_from_NN = cv2.resize(map2_from_NN, (width, height))
    map3_from_NN = cv2.resize(map3_from_NN, (width, height))
    map4_from_NN = cv2.resize(map4_from_NN, (width, height))

    map1_from_NN_previous = cv2.resize(map1_from_NN_previous, (width, height))
    map2_from_NN_previous = cv2.resize(map2_from_NN_previous, (width, height))
    map3_from_NN_previous = cv2.resize(map3_from_NN_previous, (width, height))
    map4_from_NN_previous = cv2.resize(map4_from_NN_previous, (width, height))

    # make blue
    map1_from_NN[:, :, 1] = 0
    map1_from_NN[:, :, 2] = 0
    map1_from_NN_previous[:, :, 1] = 0
    map1_from_NN_previous[:, :, 2] = 0

    # make green
    map2_from_NN[:, :, 0] = 0
    map2_from_NN[:, :, 2] = 0
    map2_from_NN_previous[:, :, 0] = 0
    map2_from_NN_previous[:, :, 2] = 0

    # make red
    map3_from_NN[:, :, 0] = 0
    map3_from_NN[:, :, 1] = 0
    map3_from_NN_previous[:, :, 0] = 0
    map3_from_NN_previous[:, :, 1] = 0

    # make yellow
    map4_from_NN[:, :, 0] = 0
    map4_from_NN_previous[:, :, 0] = 0

    print("--------after resize----------")

    print('map1_from_NN.shape=', map1_from_NN.shape)
    print('map2_from_NN.shape=', map2_from_NN.shape)
    print('map3_from_NN.shape=', map3_from_NN.shape)
    print('map4_from_NN.shape=', map4_from_NN.shape)
    print('image_for_NN.shape=', image_for_NN.shape)

    im1 = image_for_NN
    im2 = image_for_NN_previous

    h, imMatches=alignImages(im1, im2)

    map1_avg=calc_Avg_map(map1_from_NN_previous, map1_from_NN, h)
    map2_avg=calc_Avg_map(map2_from_NN_previous, map2_from_NN, h)
    map3_avg=calc_Avg_map(map3_from_NN_previous, map3_from_NN, h)
    map4_avg=calc_Avg_map(map4_from_NN_previous, map4_from_NN, h)

    #result = map1_from_NN + map2_from_NN + map3_from_NN + map4_from_NN
    result = map1_avg + map2_avg + map3_avg + map4_avg
    # cv2.imshow('result', result)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    frame_with_lanes = cv2.add(image_for_NN, result.astype('uint8'))
    # result = cv2.addWeighted(image_for_NN, 0.7, result, 0.3, 0)
    # result = result+image_for_NN
    # cv2.imshow('result', result)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return frame_with_lanes


def main():
    for i in range(1331, 1830):
        print('i=', i)

        path_to_image_for_NN = '../dataset_for_NN/Frame_' + str(i) + '.png'

        path_to_map1_from_NN = '../from_nn/images/Frame_' + str(i) + '_1_avg.png'
        path_to_map2_from_NN = '../from_nn/images/Frame_' + str(i) + '_2_avg.png'
        path_to_map3_from_NN = '../from_nn/images/Frame_' + str(i) + '_3_avg.png'
        path_to_map4_from_NN = '../from_nn/images/Frame_' + str(i) + '_4_avg.png'

        if i == 1331:
            path_to_image_for_NN_previous = '../dataset_for_NN/Frame_' + str(i) + '.png'

            path_to_map1_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_1_avg.png'
            path_to_map2_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_2_avg.png'
            path_to_map3_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_3_avg.png'
            path_to_map4_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_4_avg.png'
        else:
            path_to_image_for_NN_previous = '../dataset_for_NN/Frame_' + str(i-1) + '.png'

            path_to_map1_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_1_avg.png'
            path_to_map2_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_2_avg.png'
            path_to_map3_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_3_avg.png'
            path_to_map4_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_4_avg.png'


        frame_with_lanes = get_frame_with_lanes(
            path_to_map1_from_NN,
            path_to_map2_from_NN,
            path_to_map3_from_NN,
            path_to_map4_from_NN,
            path_to_image_for_NN,
            path_to_map1_from_NN_previous,
            path_to_map2_from_NN_previous,
            path_to_map3_from_NN_previous,
            path_to_map4_from_NN_previous,
            path_to_image_for_NN_previous)
        # write the  frame
        out.write(frame_with_lanes)

        cv2.imshow('frame_with_lanes', frame_with_lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


main()
