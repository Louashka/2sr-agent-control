from flirpy.camera.lepton import Lepton
from skimage import morphology
import numpy as np
import cv2

camera = Lepton()


while True:
    image = camera.grab()
    image = image.astype(int)

    # print(image)
    image = np.array(image)
    img = np.array(image)
    tem_max = img.max()
    print("Maximum temperature:", tem_max / 100 - 273)

    # rescale to 8 bit
    img_new = img - img.min()
    img_deno: object = img.max() - img.min()
    img = 255 * (img_new / img_deno)
    # img = 255 * (img - img.min()) / (img.max - img.min())
    # print(img)
    print(img.shape)
    length, width = img.shape

    # apply colormap
    img_col = cv2.applyColorMap(img.astype(np.uint8), cv2.COLORMAP_INFERNO)

    # Binary_map
    ret, dst = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)

    temp = list()
    # get the temperature
    for i in range(length):
        for j in range(width):
            if dst[i][j] == 255:
                temp.append([i, j, image[i][j] / 100 - 273])
    print("temperatures", temp)

    # get shape
    kernel = np.ones((2, 2), np.uint8)
    dilation = cv2.dilate(dst, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    # cv2.imshow("image", erosion)
    # gauss = cv2.GaussianBlur(erosion, (3, 3), 5)

    binary = dilation
    binary[binary == 255] = 1
    skel, distance = morphology.medial_axis(binary, return_distance=True)
    dist_on_skel = distance * skel
    dist_on_skel = dist_on_skel.astype(np.uint8) * 255
    # print(dist_on_skel)
    # get contours
    contours = list()
    for i in range(length):
        for j in range(width):
            if dist_on_skel[i][j] != 0:
                contours.append([i, j])
    print("contours", contours)

    # # check contours
    # copy = np.ones((length, width))*0
    # for contour in range(len(contours)):
    #     for i in range(length):
    #         for j in range(width):
    #             if i == contours[contour][0] and j == contours[contour][1]:
    #                 copy[i][j] = 255

    # print(image*0.04)
    # print(camera.frame_count)
    # print(camera.frame_mean)
    # print(camera.ffc_temp_k)
    # print(camera.fpa_temp_k)

    cv2.namedWindow("Binary", 0)
    cv2.resizeWindow("Binary", 300, 300)
    cv2.imshow("Binary", dst)

    cv2.namedWindow("Thermal", 0)
    cv2.resizeWindow("Thermal", 300, 300)
    cv2.imshow("Thermal", img_col)

    cv2.namedWindow("Processed", 0)
    cv2.resizeWindow("Processed", 300, 300)
    cv2.imshow("Processed", dilation)

    cv2.namedWindow("Medial_axis", 0)
    cv2.resizeWindow("Medial_axis", 300, 300)
    cv2.imshow("Medial_axis", dist_on_skel)

    # for checking contours
    # cv2.namedWindow("Contours", 0)
    # cv2.resizeWindow("Contours", 300, 300)
    # cv2.imshow("Contours", copy)

    if cv2.waitKey(1) == 27:
        break

camera.close()
cv2.destroyAllWindows()
