import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np

# Camera
fx = 1.18346606e+03
fy = 1.18757422e+03
cx = 3.14407234e+02
cy = 2.38823696e+02
k1 = -0.51328742
k2 = 0.33232725
p1 = 0.01683581
p2 = -0.00078608
k3 = -0.1159959
cameraMatrix = np.array([[fx, 0, cx],
                         [0, fy, cy],
                         [0, 0, 1]])
dist = np.array([k1, k2, p1, p2, k3])

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)
align = rs.align(rs.stream.color)


def DetectArucoPose():
    global pipeline, align
    frames = pipeline.wait_for_frames()
    # aligned_frames = align.process(frames)
    # profile = frames.get_profile()
    # depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # get the width and height
    color_image = np.asanyarray(color_frame.get_data())

    h1, w1 = color_image.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (h1, w1), 0, (h1, w1))
    frame = cv2.undistort(color_image, cameraMatrix, dist, None, newCameraMatrix)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix, dist)

        for i in range(rvec.shape[0]):
            aruco.drawAxis(frame, cameraMatrix, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners, ids)

    return frame


# draw the contours
# input1：winName
# input2：image
# input3：contours
# input4：draw_on_blank：True:drawing on a white background，False:draw on the image
def drawMyContours(WinName, Image, Contours, draw_on_blank):
    # cv2.drawContours(image, contours, index, color, line_width)
    # 输入参数：
    # image:与原始图像大小相同的画布图像（也可以为原始图像）
    # contours：轮廓（python列表）
    # index：轮廓的索引（当设置为-1时，绘制所有轮廓）
    # color：线条颜色，
    # line_width：线条粗细
    # 返回绘制了轮廓的图像image
    if draw_on_blank:  # 在白底上绘制轮廓
        temp = np.ones(Image.shape, dtype=np.uint8) * 255
        cv2.drawContours(temp, Contours, -1, (0, 1, 0), 2)
    else:
        temp = Image.copy()
        cv2.drawContours(temp, Contours, -1, (0, 1, 255), 2)


#  delete target contours
#  输入 1：contours：原始轮廓
#  输入 2：delete_list：待删除轮廓序号列表
#  返回值：contours：筛选后轮廓
def delet_contours(contours, delete_list):
    delta = 0
    contours = list(contours)
    for i in range(len(delete_list)):
        # print("i= ", i)
        del contours[delete_list[i] - delta]
        delta = delta + 1
    tuple(contours)
    return contours


def DetectCenterPoint():
    global pipeline, align
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()

    color_image = np.asanyarray(color_frame.get_data())

    # get the width and height
    height, width, channel = color_image.shape
    color_image = cv2.resize(color_image, (int(1 * width), int(1 * height)), interpolation=cv2.INTER_CUBIC)

    # 2. extract the target color from the image (in this case, yellow)
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    low_hsv = np.array([28, 222, 160])
    high_hsv = np.array([178, 255, 255])
    mask = cv2.inRange(hsv, lowerb=low_hsv, upperb=high_hsv)
    # cv2.imshow("find_yellow", mask)

    # 3. erode
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 设置kernel卷积核为 3 * 3 正方形，8位uchar型，全1结构元素
    mask = cv2.erode(mask, kernel, 3)
    # cv2.imshow("morphology", mask)

    # 4. count contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # print("find", len(contours), "contours")

    # 5. draw the contours
    drawMyContours("find contours", color_image, contours, True)

    # 6.filter for contours' length
    lengths = list()
    for i in range(len(contours)):
        length = cv2.arcLength(contours[i], True)
        lengths.append(length)
        # print("轮廓%d 的周长: %d" % (i, length))

    # use the length filter
    min_size = 30
    max_size = 80
    delete_list = []
    for i in range(len(contours)):
        if (cv2.arcLength(contours[i], True) < min_size) or (cv2.arcLength(contours[i], True) > max_size):
            delete_list.append(i)
    contours = delet_contours(contours, delete_list)
    # print("find", len(contours), "contours left after length filter")
    drawMyContours("contours after length filtering", color_image, contours, False)

    # 8.Mark the center point
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        draw_rect = cv2.drawContours(color_image.copy(), [box], -1, (0, 0, 255), 2)

        # 获取顶点坐标
        left_point_x = np.min(box[:, 0])
        right_point_x = np.max(box[:, 0])
        top_point_y = np.min(box[:, 1])
        bottom_point_y = np.max(box[:, 1])

        left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
        right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
        top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
        bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0]

        # 中央坐标值
        center_point_x = (left_point_x + right_point_x) / 2
        center_point_y = (top_point_y + bottom_point_y) / 2
        center_point = np.int0([center_point_x, center_point_y])
        # 画绿点
        circle = cv2.circle(draw_rect.copy(), center_point, 2, (0, 255, 0), 2)
        text = "(" + str(center_point[0]) + ", " + str(center_point[1]) + ")"
        all = cv2.putText(circle.copy(), text, (center_point[0] + 10, center_point[1] + 10),
                          cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 1, 8, 0)
        image = all
        cv2.imshow("centerPoint", image)

        return center_point


while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break
    image = DetectArucoPose()
    height, width, channel = image.shape
    imageCopy = cv2.resize(image, (int(1 * width), int(1 * height)), interpolation=cv2.INTER_CUBIC)
    centerPoint = DetectCenterPoint()
    circle = cv2.circle(imageCopy, centerPoint, 2, (0, 255, 0), 2)
    print(centerPoint)
    cv2.imshow("color_image", imageCopy)

pipeline.stop()
cv2.destroyAllWindows()
