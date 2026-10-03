import cv2

# Check OpenCV version
cv_version = cv2.__version__.split('.')[0]

if cv_version == '2':
    CAP_ANY = -1
    CAP_PROP_FPS = cv2.cv.CV_CAP_PROP_FPS
    CAP_PROP_FRAME_WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
else:
    CAP_ANY = cv2.CAP_ANY
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
    CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT

def create_background_subtractor():
    if cv_version == '2':
        return cv2.BackgroundSubtractorMOG2()
    else:
        return cv2.createBackgroundSubtractorMOG2()

def find_contours(image, mode, method):
    if cv_version == '2':
        contours, hierarchy = cv2.findContours(image, mode, method)
    elif cv_version == '3':
        _, contours, hierarchy = cv2.findContours(image, mode, method)
    else:
        contours, hierarchy = cv2.findContours(image, mode, method)
    return contours, hierarchy
