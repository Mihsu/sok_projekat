import cv2
import imutils
import numpy
import matplotlib.pyplot as plt


def count_people(video_path):
    # initialize values
    init_frame = None
    counted_people = 0

    x_start_plateau = 180
    plateau_width = 330
    y_start_plateau = 120
    plateau_height = 340

    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        flag, video_frame = video.read()
        # if not flag:
        #   break
        resized_frame = define_region_of_interest(video_frame, x_start_plateau, y_start_plateau, plateau_width,
                                                  plateau_height)

        grayscale_frame = to_gray_scale_frame(resized_frame)

        gb_frame = do_gaussian_blur_frame(grayscale_frame)

        if init_frame is None:
            init_frame = gb_frame
            continue

        distance = cv2.absdiff(init_frame, gb_frame)

        _, threshold_frame = cv2.threshold(distance, 21, 255, cv2.THRESH_BINARY)
        dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)

        _, contours, _ = cv2.findContours(dilated_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
            x, y, width, height = cv2.boundingRect(contour)

            cv2.rectangle(resized_frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            center_point_x = x + (width / 2)
            center_point_y = y + (height / 2)
            center_point_x_int = x + (width // 2)
            center_point_y_int = y + (height // 2)

            center_point = (center_point_x_int, center_point_y_int)
            cv2.circle(resized_frame, center_point, 1, (0, 0, 255), 2)

            plt.imshow(dilated_frame)
            plt.show()

    return counted_people


def do_gaussian_blur_frame(frame):
    return cv2.GaussianBlur(frame, (21, 21), 0)


def to_gray_scale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def define_region_of_interest(frame, x, y, width, height):
    roi = frame[y:(y + height), x:(x + width)]
    return roi


# koordinatni sistem ide od gore levo
def crossed_threshold(x, y):
    threshold = 255
    res = y - threshold
    if abs(res) <= 1:
        return True
    return False


if __name__ == '__main__':
    count_people("Videos/video1.mp4")
    print('nothing for now')
