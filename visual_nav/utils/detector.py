import argparse
import cv2
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt


class HOGDetector(object):
    def __init__(self):
        # initialize the HOG descriptor/person detector
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image, render=False):
        orig = image.copy()
        # detect people in the image
        (rects, weights) = self.hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        if render:
            # draw the final bounding boxes
            for (xA, yA, xB, yB) in pick:
                cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

            plt.imshow(image)
            plt.show()
        return pick


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="image path")
    args = ap.parse_args()

    detector = HOGDetector()
    image = cv2.imread(args.image)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    detector.detect(image)


if __name__ == '__main__':
    main()
