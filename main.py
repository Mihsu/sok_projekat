import cv2
import imutils
import numpy
import matplotlib.pyplot as plt


def count_people(video_path):
    # initialize values
    init_frame = None
    counted_people = 0

    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        flag, video_frame = video.read()

        frame = define_region_of_interest(video_frame, 405, 250, 600, 750)
        plt.show(frame)
        # if not flag:
        break

        # resized_frame = imutils.resize(video_frame, 815)
    return counted_people


def define_region_of_interest(frame, x, y, width, height):
    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
    roi = frame[y:y + height, x:x + width]

    return roi


# koordinatni sistem ide od gore levo
def crossed_threshold(x, y):
    threshold = 255
    res = y - threshold
    if abs(res) <= 1:
        return True
    return False


if __name__ == '__main__':
    print('nothing for now')
