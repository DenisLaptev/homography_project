import numpy as np
import cv2

# cap = cv2.VideoCapture(0)

# Get the Default resolutions
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))

# Define the codec and filename.
out = cv2.VideoWriter('video_with_AVERAGED_colored_lanes.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))


def get_frame_with_lanes(path_to_map1_from_NN, path_to_map2_from_NN,
                         path_to_map3_from_NN, path_to_map4_from_NN,
                         path_to_image_for_NN,
                         path_to_map1_from_NN_previous,
                         path_to_map2_from_NN_previous,
                         path_to_map3_from_NN_previous,
                         path_to_map4_from_NN_previous):
    map1_from_NN = cv2.imread(path_to_map1_from_NN)
    map2_from_NN = cv2.imread(path_to_map2_from_NN)
    map3_from_NN = cv2.imread(path_to_map3_from_NN)
    map4_from_NN = cv2.imread(path_to_map4_from_NN)

    map1_from_NN_previous = cv2.imread(path_to_map1_from_NN_previous)
    map2_from_NN_previous = cv2.imread(path_to_map2_from_NN_previous)
    map3_from_NN_previous = cv2.imread(path_to_map3_from_NN_previous)
    map4_from_NN_previous = cv2.imread(path_to_map4_from_NN_previous)

    image_for_NN = cv2.imread(path_to_image_for_NN)

    print('map1_from_NN.shape=', map1_from_NN.shape)
    print('map2_from_NN.shape=', map2_from_NN.shape)
    print('map3_from_NN.shape=', map3_from_NN.shape)
    print('map4_from_NN.shape=', map4_from_NN.shape)
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

    map1 = map1_from_NN.astype(float)/2+map1_from_NN_previous.astype(float)/2
    map1 = map1.astype(int)

    map2 = map2_from_NN.astype(float) / 2 + map2_from_NN_previous.astype(float) / 2
    map2 = map2.astype(int)

    map3 = map3_from_NN.astype(float) / 2 + map3_from_NN_previous.astype(float) / 2
    map3 = map3.astype(int)

    map4 = map4_from_NN.astype(float) / 2 + map4_from_NN_previous.astype(float) / 2
    map4 = map4.astype(int)

    #result = map1_from_NN + map2_from_NN + map3_from_NN + map4_from_NN
    result = map1 + map2 + map3 + map4
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

        path_to_map1_from_NN = '../from_nn/images/Frame_' + str(i) + '_1_avg.png'
        path_to_map2_from_NN = '../from_nn/images/Frame_' + str(i) + '_2_avg.png'
        path_to_map3_from_NN = '../from_nn/images/Frame_' + str(i) + '_3_avg.png'
        path_to_map4_from_NN = '../from_nn/images/Frame_' + str(i) + '_4_avg.png'

        if i == 1331:
            path_to_map1_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_1_avg.png'
            path_to_map2_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_2_avg.png'
            path_to_map3_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_3_avg.png'
            path_to_map4_from_NN_previous = '../from_nn/images/Frame_' + str(i) + '_4_avg.png'
        else:
            path_to_map1_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_1_avg.png'
            path_to_map2_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_2_avg.png'
            path_to_map3_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_3_avg.png'
            path_to_map4_from_NN_previous = '../from_nn/images/Frame_' + str(i - 1) + '_4_avg.png'

        path_to_image_for_NN = '../dataset_for_NN/Frame_' + str(i) + '.png'

        frame_with_lanes = get_frame_with_lanes(
            path_to_map1_from_NN,
            path_to_map2_from_NN,
            path_to_map3_from_NN,
            path_to_map4_from_NN,
            path_to_image_for_NN,
            path_to_map1_from_NN_previous,
            path_to_map2_from_NN_previous,
            path_to_map3_from_NN_previous,
            path_to_map4_from_NN_previous)
        # write the  frame
        out.write(frame_with_lanes)

        cv2.imshow('frame_with_lanes', frame_with_lanes)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()


main()
