import numpy as np
import cv2

cap = cv2.VideoCapture('../solution4_video_lanes/video_with_colored_lanes.avi')
cap_avg = cv2.VideoCapture('../solution4_video_lanes/video_with_AVERAGED_colored_lanes.avi')

# Define the codec and filename.
out_2videos_diff = cv2.VideoWriter('initial_and_avg_DIFFERENCE.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (1280, 720))
out_2videos = cv2.VideoWriter('initial_and_avgeraged.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (640, 720))


def get_difference(frame_initial, frame_averaged):

    frame_initial_float = frame_initial.astype(float)
    frame_averaged_float = frame_averaged.astype(float)
    print('frame_initial_float=', frame_initial_float)
    print('frame_averaged_float=', frame_averaged_float)

    frames_difference_float = frame_initial_float - frame_averaged_float
    print('frames_difference_float=', frames_difference_float)

    frames_difference_float_abs = np.absolute(frames_difference_float)
    print('frames_difference_float_abs=', frames_difference_float_abs)

    frames_difference_int_abs = frames_difference_float_abs.astype(int)
    print('frames_difference_int_abs=', frames_difference_int_abs)
    # cv2.imshow('frames_difference_int_abs', frames_difference_int_abs.astype('uint8'))

    frames_maximum = np.maximum(frame_initial, frame_averaged)
    # cv2.imshow('frames_maximum', frames_maximum)

    return frames_difference_int_abs.astype('uint8')


frame_count = 0

while (cap.isOpened()):

    ret, frame = cap.read()
    ret_avg, frame_avg = cap_avg.read()

    if ret==True and ret_avg==True:

        height, width, _ = frame.shape
        print('frame.shape=', frame.shape)

        cv2.imshow('frame', frame)
        cv2.imshow('frame_avg', frame_avg)

        frames_difference = get_difference(frame_initial=frame,
                                           frame_averaged=frame_avg)

        cv2.imshow('frames_difference', frames_difference)
        out_2videos_diff.write(frames_difference)


        frame_resized=cv2.resize(src=frame,dsize=(width//2,height//2))
        frame_avg_resized=cv2.resize(src=frame_avg,dsize=(width//2,height//2))
        initial_and_averaged = cv2.vconcat([frame_resized, frame_avg_resized])

        cv2.imshow('initial_and_averaged', initial_and_averaged)
        out_2videos.write(initial_and_averaged)

        frame_count = frame_count + 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('FINISH:frame_count=', frame_count)
            break

    else:
        break

print('number_of_frames=', frame_count+1)

cap.release()
cap_avg.release()

out_2videos_diff.release()
out_2videos.release()

cv2.destroyAllWindows()
