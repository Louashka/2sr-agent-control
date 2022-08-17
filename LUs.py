import pyrealsense2 as rs
import cv2
import cv2.aruco as aruco
import numpy as np
import math


class LU:
    # segments
    l = 0.04  # 40 mm in meter
    # Camera
    fx = 461.84448242
    fy = 443.28289795
    cx = 308.69522309
    cy = 177.70244623
    k1 = 0.04266696
    k2 = -0.11292418
    p1 = 0.00306782
    p2 = -0.00409565
    k3 = 0.02006348
    cameraMatrix = np.array([[fx, 0, cx],
                             [0, fy, cy],
                             [0, 0, 1]])
    dist = np.array([k1, k2, p1, p2, k3])

    def start(self):
        # Configure depth and color streams
        config = rs.config()
        # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline = rs.pipeline()
        self.pipeline.start(config)

    def stop(self):
        self.pipeline.stop()

    def show_image(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        # get the width and height
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("image", color_image)

    def cal_Points(self, Ids, Corners, rvec, tvec):
        if Ids is not None:
            center = [[0., 0.], [0., 0.], [0., 0.]]
            rvecs = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
            tvecs = [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]]
            for i in range(len(Ids)):
                for i in range(len(Ids)):
                    if Ids[i] == 1:
                        corner_A = Corners[i][0]
                        center_x = (corner_A[0][0] + corner_A[2][0]) / 2
                        center_y = (corner_A[0][1] + corner_A[2][1]) / 2
                        center[1][0] = center_x
                        center[1][1] = center_y
                        rvecs[1] = rvec[i][0]
                        tvecs[1] = tvec[i][0]
                        # print("centers", centers)
                        # print("rvecs", rvecs)
                    elif Ids[i] == 2:
                        corner_B = Corners[i][0]
                        center_x = (corner_B[0][0] + corner_B[2][0]) / 2
                        center_y = (corner_B[0][1] + corner_B[2][1]) / 2
                        center[2][0] = center_x
                        center[2][1] = center_y
                        rvecs[2] = rvec[i][0]
                        tvecs[2] = tvec[i][0]
                    elif Ids[i] == 0:
                        corner_Center = Corners[i][0]
                        center_x = (corner_Center[0][0] + corner_Center[2][0]) / 2
                        center_y = (corner_Center[0][1] + corner_Center[2][1]) / 2
                        center[0][0] = center_x
                        center[0][1] = center_y
                        rvecs[0] = rvec[i][0]
                        tvecs[0] = tvec[i][0]
            # print(edges)
            center = np.int0(center)
            # circle = cv2.circle(image.copy(), center[0], 2, (0, 255, 0), 2)
            # circle = cv2.circle(circle(), center[1], 2, (0, 255, 0), 2)
            # circle = cv2.circle(circle(), center[2], 2, (0, 255, 0), 2)
            # cv2.imshow("image", circle)
            return center, rvecs
            print("realsense", realsense.center)

    def DetectArucoPose(self):
        frames = self.pipeline.wait_for_frames()
        # aligned_frames = align.process(frames)
        # profile = frames.get_profile()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # get the width and height
        color_image = np.asanyarray(color_frame.get_data())

        h1, w1 = color_image.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(LU.cameraMatrix, LU.dist, (h1, w1), 0, (h1, w1))
        frame = cv2.undistort(color_image, LU.cameraMatrix, LU.dist, None, newCameraMatrix)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, LU.cameraMatrix, LU.dist)
            # print("rvec: ", rvec)
            # print("tvex: ", tvec)
            for i in range(rvec.shape[0]):
                tvecCopy = tvec[i, :, :] + [10., 0, 0]
                # print("tvecCopy", tvecCopy)
                aruco.drawAxis(frame, LU.cameraMatrix, LU.dist, rvec[i, :, :], tvec[i, :, :], 0.03)
                aruco.drawDetectedMarkers(frame, corners, ids)
                # print("rvec[", i, ",: , ：]: ", rvec[i, :, :])
            cv2.imshow("arucoDetector", frame)
            centers, rotationVec = self.cal_Points(ids, corners, rvec, tvec)
            return centers, rotationVec


lu = LU()
lu.start()
while True:
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        lu.stop()
        break
    centers, rvecs = lu.DetectArucoPose()
    print("centers", centers)
    print("rvecs", rvecs)