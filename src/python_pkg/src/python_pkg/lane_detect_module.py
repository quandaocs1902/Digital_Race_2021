import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import numpy as np
import sys
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from numpy.lib.polynomial import polyfit
from sklearn.metrics import mean_squared_error

class PolynomialRegression(object):
    def __init__(self, degree = 3, coeffs = None):
        self.degree = degree
        self.coeffs = coeffs
    
    def fit(self, X, y):
        self.coeffs = np.polyfit(X.ravel(), y, self.degree)

    def get_params(self, deep = False):
        return {'coeffs': self.coeffs}
    
    def set_params(self, coeffs = None, random_state = None):
        self.coeffs = coeffs
    
    def predict(self, X):
        poly_eqn = np.poly1d(self.coeffs)
        y_hat = poly_eqn(X.ravel())
        return y_hat
    
    def score(self, X, y):
        return mean_squared_error(y, self.predict(X))

def pers_transform(img):
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[0, 240], [160 - 640/21, 100],
                     [160 + 640/21, 100], [320, 240]])

    cv2.imshow("img", img)
    # cv2.waitKey(2)
    offset = [80, 0]
    dst = np.float32([src[0] + offset, np.array([src[0, 0], 0]) + offset,
                      np.array([src[3, 0], 0]) - offset, src[3] - offset])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    # Return the resulting image and matrix
    return warped, M
# Find average threshold of line
def find_threshold(img):
    roi = img[160:190, 140:180]
    threshold = np.mean(roi)
    return threshold

# Data preprocessing
def preprocess(img):
    # Read image
    global warped
    # img = cv.imread(str(image_path), cv.IMREAD_COLOR)
    warped, _ = pers_transform(img)
    # Convert in to HSV color space
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    # Blurr image
    s_blurred = cv2.medianBlur(s, 3)
    v_blurred = cv2.medianBlur(v, 3)

    # Convert to binary image
    s_threshold = find_threshold(s) + 20
    v_threshold = find_threshold(v) + 50

    _, s_thresh = cv2.threshold(
        s_blurred, s_threshold, 255, cv2.THRESH_BINARY_INV)
    _, v_thresh = cv2.threshold(v_blurred, v_threshold, 255, cv2.THRESH_BINARY)

    # final = cv.bitwise_and(s_thresh, v_thresh)[100:, :]
    final = cv2.bitwise_and(s_thresh, v_thresh)
    #canny = canny_edge_detect(final)
    canny = cv2.Canny(final, 100, 200)
    process = downsample(canny)
    left, right = getmeanX(process, img)
    try:
        leftCurve = ransacSklearn(left)
        cv2.polylines(img, [leftCurve], False, (255, 0, 0), 2)
    except IndexError:
        pass
    try:
        rightCurve = ransacSklearn(right)
        cv2.polylines(img, [rightCurve], False, (255, 0, 0), 2)
    except IndexError:
        pass
    return img

def downsample(ROI):
    col_index = 0
    result = np.zeros((24, 320), dtype=np.float64)
    for i in range(0, ROI.shape[0], 10):
        for j in range(0, ROI.shape[1]):
            process = ROI[i: i + 10, j]
            value = np.sum(process, dtype=np.float64)
            result[col_index][j] = value
        col_index += 1
    return result

# Get consecutive array
def ranges(nums):
    gaps = [[s, e]
            for s, e in zip(nums, nums[1:]) if (s + 1 < e and s + 70 < e)]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

# Determine xLeft, xMid, xRight
def getmeanX(process, img):
    dataLeft = []
    dataRight = []
    dataMid = []
    for i in range(process.shape[0]):
        ROI = process[i, :]
        xValue = np.where(ROI > 0)
        x = ranges(list(xValue[0]))
        try:
            for j in x:
                if (np.mean(j, dtype=int) <= 80):
                    left = [np.mean(j, dtype=int), 5 + i * 10]
                    dataLeft.append(left)
                    # Tuple to draw circle
                    l = (np.mean(j, dtype = int), 5 + i * 10)
                    cv2.circle(warped, l, 2, (214, 57, 17), 2)
                elif (np.mean(j, dtype=int) >= 160):
                    right = [np.mean(j, dtype=int), 5 + i * 10]
                    dataRight.append(right)
                    # Tuple to draw circle
                    r = (np.mean(j, dtype = int), 5 + i * 10)
                    cv2.circle(warped, r, 2, (0, 84, 163), 2)
                else:
                    dataMid.append((np.mean(j, dtype=int), 5 + i * 10))
        except IndexError:
            pass
        cv2.imshow("warped", warped)
        cv2.waitKey(2)
    return [np.asarray(dataLeft, dtype=int), np.asarray(dataRight, dtype=int)]

# Using Ransac to fit lane
def ransacSklearn(data):
    data = np.sort(data, axis = 0)
    X = np.asarray(data[: , 0], dtype = np.int32)
    y = data[: , 1]

    lineX = X.reshape((len(data), 1))
    ransac = RANSACRegressor(PolynomialRegression(degree = 2), residual_threshold= 2 * np.std(y), random_state=0)
    ransac.fit(lineX, y)
    lineY = ransac.predict(lineX)
    poly_Coef = np.polyfit(X, lineY, 2)
    lineY = np.polyval(poly_Coef, lineX).reshape((-1, 1))
    result = np.hstack((lineX, lineY))
    return np.int32(result)
