import cv2
import imutils
import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


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
        if not flag:
            break
        resized_frame = define_region_of_interest(video_frame, x_start_plateau, y_start_plateau, plateau_width,
                                                  plateau_height)

        grayscale_frame = to_gray_scale_frame(resized_frame)

        gb_frame = do_gaussian_blur_frame(grayscale_frame)

        if init_frame is None:
            init_frame = gb_frame
            continue

        distance = cv2.absdiff(init_frame, gb_frame)

        _, threshold_frame = cv2.threshold(distance, 13, 255, cv2.THRESH_BINARY)
        dilated_frame = cv2.dilate(threshold_frame, None, iterations=2)

        _, contours, _ = cv2.findContours(dilated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 120:
                continue
            x, y, width, height = cv2.boundingRect(contour)

            cv2.rectangle(resized_frame, (x, y), (x + width, y + height), (255, 0, 0), 2)

            center_point_x = x + (width / 2)
            center_point_y = y + (height / 2)
            center_point_x_int = x + (width // 2)
            center_point_y_int = y + (height // 2)

            center_point = (center_point_x_int, center_point_y_int)
            cv2.circle(resized_frame, center_point, 1, (0, 0, 255), 2)

            cv2.line(resized_frame, (0, plateau_height // 3 ), (plateau_width, plateau_height // 3), (0, 0, 255), 2)
            if crossed_threshold(center_point_y, plateau_height / 3):
                counted_people += 1

            # plt.imshow(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            # plt.show()
    print(counted_people)
    return counted_people


def return_mae(given, final):
    mae = mean_absolute_error(given, final)
    return mae


def do_gaussian_blur_frame(frame):
    return cv2.GaussianBlur(frame, (21, 21), 0)


def to_gray_scale_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def define_region_of_interest(frame, x, y, width, height):
    roi = frame[y:(y + height), x:(x + width)]
    return roi


# koordinatni sistem ide od gore levo
def crossed_threshold(y, y_center_frame):
    res = y - y_center_frame
    if abs(res) <= 1:
        return True
    return False


def load_results():
    results = [count_people("Videos/video1.mp4"), count_people("Videos/video2.mp4"),
               count_people("Videos/video3.mp4"), count_people("Videos/video4.mp4"),
               count_people("Videos/video5.mp4"), count_people("Videos/video6.mp4"),
               count_people("Videos/video7.mp4"), count_people("Videos/video8.mp4"),
               count_people("Videos/video9.mp4"), count_people("Videos/video10.mp4")]
    return results


if __name__ == '__main__':
    given_results = [4, 24, 17, 23, 17, 27, 29, 22, 10, 23]
    final_results = load_results()

    print("Mae:")
    print(str(return_mae(given_results, final_results)))
