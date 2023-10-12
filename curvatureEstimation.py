import numpy as np
import pandas as pd
# import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import glob
import cv2
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, splprep, splev
import scipy.signal
import re
import ast
from pathlib import Path
from skimage import morphology


path = os.getcwd()
file_pattern = re.compile(r'.*?(\d+).*?')
coef = 0.0002380952380952381

def sort_files(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def read_images():
    path_neg = os.path.join(path + '/ExpData/TempPhotos/Negative/', '*.jpg')
    img_neg_files = sorted(glob.glob(path_neg), key = sort_files)

    path_pos = os.path.join(path + '/ExpData/TempPhotos/Positive/', '*.jpg')
    img_pos_files = sorted(glob.glob(path_pos), key = sort_files)

    images_neg = [cv2.imread(file) for file in img_neg_files]
    images_pos = [cv2.imread(file) for file in img_pos_files]

    images = images_neg + images_pos
    images_resized = []

    for img in images:
        scale = 20
        images_resized.append(cv2.resize(img, None, fx = scale, fy = scale, interpolation= cv2.INTER_CUBIC))

    return images_resized

def read_csv(images):
    csv_neg = glob.glob(os.path.join(path + '/ExpData/', 'exp_negative_curvature.csv'))
    csv_pos = glob.glob(os.path.join(path + '/ExpData/', 'exp_positive_curvature.csv'))

    df_neg = pd.read_csv(csv_neg[0], index_col = None, header = 0)
    df_pos = pd.read_csv(csv_pos[0], index_col = None, header = 0)

    df = pd.concat([df_neg, df_pos.iloc[1:]], ignore_index=True)

    curves = extract_curves(images)
    df['curve'] = curves

    df.to_csv('ExpData/exp_curvature.csv', index=False)

    return df


def extract_curves(images):
    curves = []

    counter = 1
    for img in images:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(img_gray, 120, 255, cv2.THRESH_BINARY)

        blur = cv2.bilateralFilter(thresh, 9, 75, 75)

        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(blur, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        dilation = cv2.dilate(erosion, kernel, iterations=1)

        binary = dilation.copy()
        binary[binary == 255] = 1

        binary_smoothed = scipy.signal.medfilt(binary, 7)

        print(counter)
        counter += 1

        skel, distance = morphology.medial_axis(binary_smoothed, return_distance=True)
        dist_on_skel = distance * skel
        dist_on_skel = dist_on_skel.astype(np.uint8) * 255

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        contours, hierarchy = cv2.findContours(image=dist_on_skel, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

        x = contours[0][:,0,0]
        y = contours[0][:,0,1]
        curve = np.column_stack((x, y)).tolist()

        # curve = get_curve(x, y)
        curves.append(curve)

    return curves


def str_to_points(st):
    arr = ast.literal_eval(st)

    return arr


def get_curvature(curve, k, smoothing = 4):

    df = pd.DataFrame()

    curve = np.array(curve)
    end_point = curve.shape[0]
    start_point = int(end_point / 2)

    x = curve[start_point:, 0] * coef
    y = -curve[start_point:, 1] * coef

    z = np.polyfit(y, x, 4)
    f = np.poly1d(z)

    y_new = y
    x_new = f(y_new)

    df['dx'] = np.gradient(x_new)
    df['dx'] = df.dx.rolling(smoothing, center = True).mean()

    df['dy'] = np.gradient(y_new)
    df['dy'] = df.dy.rolling(smoothing, center = True).mean()

    df['d2x'] = np.gradient(df.dx)
    df['d2x'] = df.d2x.rolling(smoothing, center = True).mean()

    df['d2y'] = np.gradient(df.dy)
    df['d2y'] = df.d2y.rolling(smoothing, center = True).mean()

    df['k'] = df.eval('abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5')
    df['k'] = df.k.rolling(smoothing, center = True).mean()
    df['k'] = df['k'].fillna(0)
    if k < 0:
        df.k = -df.k

    k_therm = df.k.to_list()

    return k_therm



if __name__ == "__main__":

    images = read_images()
    df = read_csv(images)

    csv_file = glob.glob(os.path.join(path + '/ExpData/', 'exp_curvature.csv'))
    df = pd.read_csv(csv_file[0], index_col = None, header = 0)
    df.curve = df.curve.apply(str_to_points)
    df['therm_k'] = df.apply(lambda row: get_curvature(row['curve'], row['k1']), axis = 1)
    # print(df.therm_k.iloc[2])
    df['therm_k_mean'] = df.apply(lambda row: np.mean(row['therm_k']), axis = 1)
    df['therm_k_std'] = df.apply(lambda row: np.std(row['therm_k']), axis = 1)

    # df.to_csv(csv_file[0], index = False)

    x = df.index.values

    plt.scatter(x, df.k1)
    plt.scatter(x, df.therm_k_mean)
    plt.show()

    # n = 131

    # test_curve = np.array(df.curve.iloc[n])

    # end_point = test_curve.shape[0]
    # start_point = int(test_curve.shape[0] / 2)

    # x = test_curve[start_point:, 0] * coef
    # y = -test_curve[start_point:, 1] * coef

    # z = np.polyfit(y, x, 4)
    # f = np.poly1d(z)

    # y_new = y
    # x_new = f(y_new)


    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_aspect('equal', adjustable='box')
    # ax.plot(x, y, '.', x_new, y_new)
    # fig.canvas.draw()

    # img_vss = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
    #                         sep='')
    # img_vss = img_vss.reshape(
    #     fig.canvas.get_width_height()[::-1] + (3,))

    # # img is rgb, convert to opencv's default bgr
    # img_vss = cv2.cvtColor(img_vss, cv2.COLOR_RGB2BGR)

    # # display image with opencv or any operation you like
    # cv2.imshow("Curve", img_vss)

    # img = images[n]

    # cv2.imshow("Thermal image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
