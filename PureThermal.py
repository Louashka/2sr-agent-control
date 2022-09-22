from flirpy.camera.lepton import Lepton
from scipy.interpolate import UnivariateSpline
from skimage import morphology
import numpy as np
import cv2


class PureThermal:
    camera = Lepton()
    flag = False

    def stop(self):
        self.camera.close()
        cv2.destroyAllWindows()

    def curvature(self, x, y, error=0.1):
        t = np.arange(x.shape[0])
        std = error * np.ones_like(x)

        fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std))
        fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std))

        xˈ = fx.derivative(1)(t)
        xˈˈ = fx.derivative(2)(t)
        yˈ = fy.derivative(1)(t)
        yˈˈ = fy.derivative(2)(t)
        curvature = (xˈ * yˈˈ - yˈ * xˈˈ) / np.power(xˈ * 2 + yˈ * 2, 3 / 2)
        theta = np.arctan(xˈ)

        return curvature, theta

    def grab_image(self, camera):
        image = self.camera.grab()
        image = image.astype(int)
        # print(camera.frame_count)
        # print(camera.frame_mean)
        # print(camera.ffc_temp_k)
        # print(camera.fpa_temp_k)

        # return the raw data
        return image

    def show_image(self, image):
        img, tem_max = self.get_image(image)

        # apply colormap
        img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)
        # show image
        cv2.namedWindow("Thermal", 0)
        cv2.resizeWindow("Thermal", 300, 300)
        cv2.imshow("Thermal", img_col)

        # return the raw data "image" and the 8-bit data "img"
        # print(img)
        return img, tem_max / 100 - 273

    def get_image(self, image):
        # print(image)
        image = np.array(image)
        img = np.array(image)
        tem_max = img.max()
        # print("Maximum temperature:", tem_max / 100 - 273)

        # rescale to 8 bit
        img_new = img - img.min()
        img_deno: object = img.max() - img.min()
        img = 255 * (img_new / img_deno)
        # img = 255 * (img - img.min()) / (img.max - img.min())
        # print(img)
        # print(img.shape)

        # return the raw data "image" and the 8-bit data "img"
        # print(img)
        return img, tem_max / 100 - 273

    def get_temperature(self, image, img):
        # Binary_map
        ret, dst = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

        # cv2.namedWindow("Binary", 0)
        # cv2.resizeWindow("Binary", 300, 300)
        # cv2.imshow("Binary", dst)

        length, width = img.shape
        temp = list()
        # only get the temperature from the interested area (high temperature)
        for i in range(length):
            for j in range(width):
                if dst[i][j] == 255:
                    temp.append([i, j, image[i][j] / 100 - 273])
        temp = np.asarray(temp)
        # print("temperatures", temp)

        # return the temperature of the target area
        return temp

    def get_contours(self, img):
        # Binary_map
        ret, dst = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
        # pre-processing for getting the shape
        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(dst, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)
        # cv2.imshow("image", erosion)

        binary = dilation
        binary[binary == 255] = 1

        # cv2.namedWindow("Processed", 0)
        # cv2.resizeWindow("Processed", 300, 300)
        # cv2.imshow("Processed", dilation)

        skel, distance = morphology.medial_axis(binary, return_distance=True)
        dist_on_skel = distance * skel
        dist_on_skel = dist_on_skel.astype(np.uint8) * 255
        # print(dist_on_skel)

        # cv2.namedWindow("Medial_axis", 0)
        # cv2.resizeWindow("Medial_axis", 300, 300)
        # cv2.imshow("Medial_axis", dist_on_skel)

        # get contours
        contours = list()
        contours_x = list()
        contours_y = list()
        length, width = img.shape
        for i in range(length):
            for j in range(width):
                if dist_on_skel[i][j] != 0:
                    contours.append([i, j])
                    contours_x.append(i)
                    contours_y.append(j)
        contours = np.asarray(contours)
        contours_x = np.asarray(contours_x)
        contours_y = np.asarray(contours_y)
        # print("contours", contours)

        # # check contours
        # copy = np.ones((length, width))*0
        # for contour in range(len(contours)):
        #     for i in range(length):
        #         for j in range(width):
        #             if i == contours[contour][0] and j == contours[contour][1]:
        #                 copy[i][j] = 255
        # # for checking contours
        # cv2.namedWindow("Contours", 0)
        # cv2.resizeWindow("Contours", 300, 300)
        # cv2.imshow("Contours", copy)

        # return the position of contours
        return contours, contours_x, contours_y

    def get_data(self):
        image = self.grab_image(PureThermal.camera)
        if image is not None:
            img, temp_max = self.get_image(image)
            temp = self.get_temperature(image, img)
            contours, x, y = self.get_contours(img)
            curve, theta = self.curvature(x, y)
            # print('contours', contours)
            # print("theta", theta / 0.04)
            curveture = theta / 0.04
            t_current = [temp_max, curveture]
            return t_current

# thermal = PureThermal()
# while True:
#     image = thermal.grab_image(thermal.camera)
#     if image is not None:
#         img, temp_max = thermal.show_image(image)
#         temp = thermal.get_temperature(image, img)
#         contours, x, y = thermal.get_contours(img)
#         curve, theta = thermal.curvature(x, y)
#         print("Maximum tempeture: ", temp_max)
#         print('contours', contours)
#         print("theta", theta / 0.04)
#
#     if cv2.waitKey(10) == 27:
#         break
# cv2.destroyAllWindows()
