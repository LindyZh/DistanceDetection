import os
import cv2
import math
import time
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import init

# ================== initialization ===================================
# testing params
cam_id = 3
sampling_rate = 5  # fps
clear = -1  # people with this numer of occluded joints will be removed

path = init.bb_data_path.format(cam_id)
calibration_path = init.cal_data_path.format(cam_id)
csv_path = os.path.join(path, "coords_fib_cam_{}.csv".format(cam_id))

# make sure the cleared cam is stored here
if clear==-1:
    annotated_csv_path = os.path.join(calibration_path, "coords_cam_{}.csv".format(cam_id))
else:
    annotated_csv_path = os.path.join(calibration_path, "clear_{}_coords_cam_{}.csv".format(clear, cam_id))

video_path = os.path.join(path, "cam_{}.mp4".format(cam_id))

# create dataframe from csv file and read video
start_time = time.time()
cam_coords = pd.read_csv(csv_path)
print("Bounding Box Data loading finished in %s seconds ---" % (time.time() - start_time))
annotated_csv = pd.read_csv(annotated_csv_path)
print("All position labelling data loading finished in %s seconds ---" % (time.time() - start_time))

# config the camera
x_cam = annotated_csv.iloc[0]["x_3D_cam"]
y_cam = annotated_csv.iloc[0]["y_3D_cam"]
z_cam = annotated_csv.iloc[0]["z_3D_cam"]

unique_sorted_y_coords = sorted(list(set(annotated_csv["y_bottom_right_BB"].to_list())))
# get pixel height vs distance from one person, ideally around center of screen
person = annotated_csv[annotated_csv.y_bottom_right_BB == unique_sorted_y_coords[len(unique_sorted_y_coords) // 2]]
x_p = person.iloc[0]["x_3D_person"]
y_p = person.iloc[0]["y_3D_person"]
dis = math.sqrt((x_cam - x_p) ** 2 + (y_cam - y_p) ** 2)
pixel_h_top = person.iloc[0]["y_top_left_BB"]
pixel_h_bottom = person.iloc[0]["y_bottom_right_BB"]
pixel_h = abs(pixel_h_top - pixel_h_bottom)
# calculate the focus length
focus_len = pixel_h * dis

# get x_dis on screen vs distance in 3d, with x_dis calculated near lower edge of screen
person1 = annotated_csv[annotated_csv.y_bottom_right_BB == unique_sorted_y_coords[0]]
person2 = annotated_csv[annotated_csv.y_bottom_right_BB == unique_sorted_y_coords[1]]
xB1 = person1.iloc[0]["x_bottom_right_BB"]
xT1 = person1.iloc[0]["x_top_left_BB"]
xC1 = (xB1 + xT1) / 2
x_3d1 = person1.iloc[0]["x_3D_person"]
y_3d1 = person1.iloc[0]["y_3D_person"]
xB2 = person2.iloc[0]["x_bottom_right_BB"]
xT2 = person2.iloc[0]["x_top_left_BB"]
xC2 = (xB2 + xT2) / 2
x_3d2 = person2.iloc[0]["x_3D_person"]
y_3d2 = person2.iloc[0]["y_3D_person"]
# get ratio
dis_3d = math.sqrt((x_3d1 - x_3d2) ** 2 + (y_3d1 - y_3d2) ** 2)
x_dis_2d = abs(xC1 - xC2)
ratio = dis_3d / x_dis_2d


# ================== helper fcn =======================================
def draw_bounding_box(image, bounding_box):
    cv2.line(image, (bounding_box.xT, bounding_box.yT), (bounding_box.xT, bounding_box.yB), init.BB_color, thickness=1)
    cv2.line(image, (bounding_box.xB, bounding_box.yT), (bounding_box.xB, bounding_box.yB), init.BB_color, thickness=1)
    cv2.line(image, (bounding_box.xT, bounding_box.yB), (bounding_box.xB, bounding_box.yB), init.BB_color, thickness=1)
    cv2.line(image, (bounding_box.xT, bounding_box.yT), (bounding_box.xB, bounding_box.yT), init.BB_color, thickness=1)
    cv2.putText(image, str(bounding_box.id), (bounding_box.xT, bounding_box.yT), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, init.pid_color, 1, cv2.LINE_AA)


# ================== BB class =========================================

class BB:
    xT = 0
    yT = 0
    xB = 0
    yB = 0
    xC = 0
    yC = 0
    id = -1

    def __init__(self, xt, yt, xb, yb, pid):
        self.xT = xt
        self.yT = yt
        self.xB = xb
        self.yB = yb
        self.xC = (xb + xt) // 2
        self.yC = (yb + yt) // 2
        self.id = pid


# ================== accuracy graph initialization ====================
observed_d = []
accurate_d = []
observed_dis = []
accurate_dis = []
observed_dis1 = []
accurate_dis1 = []
observed_dis2 = []
accurate_dis2 = []
# ================== creating video data ==============================

count = 0
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("total frame num: ", length)
while cap.isOpened():
    # to avoid bug for size of -1 at the last frame
    if count == length - 2:
        break

    ret, frame = cap.read()
    if count % sampling_rate == 0:
        bbox_list = []
        accurate_dict = {}
        xT = cam_coords[cam_coords.frame_no_cam == count]["x_top_left_BB"].to_list()
        yT = cam_coords[cam_coords.frame_no_cam == count]["y_top_left_BB"].to_list()
        xB = cam_coords[cam_coords.frame_no_cam == count]["x_bottom_right_BB"].to_list()
        yB = cam_coords[cam_coords.frame_no_cam == count]["y_bottom_right_BB"].to_list()
        p_ID = cam_coords[cam_coords.frame_no_cam == count]["person_id"].to_list()

        # for depth precision checking
        xP = annotated_csv[(annotated_csv.frame_no_cam == count) &
                           (annotated_csv.joint_type == 0)]["x_3D_person"].to_list()
        yP = annotated_csv[(annotated_csv.frame_no_cam == count) &
                           (annotated_csv.joint_type == 0)]["y_3D_person"].to_list()
        p_id_annotated = annotated_csv[(annotated_csv.frame_no_cam == count) &
                                       (annotated_csv.joint_type == 0)]["person_id"].to_list()

        for i in range(len(p_id_annotated)):
            accurate_dict[p_id_annotated[i]] = [xP[i], yP[i]]

        for i in range(len(xT)):
            bbox = BB(xT[i], yT[i], xB[i], yB[i], p_ID[i])
            draw_bounding_box(frame, bbox)
            bbox_list.append(bbox)

        if len(bbox_list) > 1:
            tar_pair = list(itertools.combinations(bbox_list, 2))

            for i in range(len(tar_pair)):
                # check distance here, if too close draw the distance line

                try:
                    label_1 = accurate_dict[tar_pair[i][0].id]
                    label_2 = accurate_dict[tar_pair[i][1].id]
                except:
                    break

                # depth accuracy testing =====================================================
                ydis1 = focus_len / (tar_pair[i][0].yB - tar_pair[i][0].yT)
                ydis2 = focus_len / (tar_pair[i][1].yB - tar_pair[i][1].yT)

                actual_dis1 = math.sqrt((label_1[0] - x_cam) ** 2 + (label_1[1] - y_cam) ** 2)
                actual_dis2 = math.sqrt((label_2[0] - x_cam) ** 2 + (label_2[1] - y_cam) ** 2)

                observed_d.append(ydis1)
                observed_d.append(ydis2)
                accurate_d.append(actual_dis1)
                accurate_d.append(actual_dis2)

                # distance accuracy testing ====================================================
                x_center = (1920 + 1) / 2
                w_to_cam_1 = tar_pair[i][0].xC - x_center
                h_to_cam_1 = 1080 - tar_pair[i][0].yB
                w_to_cam_2 = tar_pair[i][1].xC - x_center
                h_to_cam_2 = 1080 - tar_pair[i][1].yB

                # old method, polar
                theta_1 = math.atan(h_to_cam_1 / w_to_cam_1)
                theta_2 = math.atan(h_to_cam_2 / w_to_cam_2)
                D = math.sqrt(ydis1 ** 2 + ydis2 ** 2 - 2 * ydis2 * ydis1 * math.cos(theta_1 - theta_2))

                actual_dis = math.sqrt((label_1[0] - label_2[0]) ** 2 + (label_1[1] - label_2[1]) ** 2)

                observed_dis.append(D)
                accurate_dis.append(actual_dis)

                # new method using theta
                x1 = w_to_cam_1 * ratio
                x2 = w_to_cam_2 * ratio
                theta_1 = math.acos(max(min(x1 / ydis1, 1), -1))
                theta_2 = math.acos(max(min(x2 / ydis2, 1), -1))
                D = math.sqrt(ydis1 ** 2 + ydis2 ** 2 - 2 * ydis2 * ydis1 * math.cos(theta_1 - theta_2))

                observed_dis1.append(D)
                accurate_dis1.append(actual_dis)

                # new method using x y
                x1 = w_to_cam_1 * ratio
                y1 = math.sqrt(max(ydis1 ** 2 - x1 ** 2, 0))
                x2 = w_to_cam_2 * ratio
                y2 = math.sqrt(max(ydis2 ** 2 - x2 ** 2, 0))

                D2 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                observed_dis2.append(D2)
                accurate_dis2.append(actual_dis)

                # check distance here, if too close draw the distance line
                if D < 5:
                    cv2.line(frame, (tar_pair[i][0].xC, tar_pair[i][0].yC), (tar_pair[i][1].xC, tar_pair[i][1].yC),
                             init.dis_color, thickness=2)
        if init.show_vid:
            cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
    count += 1

cap.release()
cv2.destroyAllWindows()

# depth plot
plt.scatter(observed_d, accurate_d, color='r')
xpoints = np.array([0, 100])
ypoints = np.array([0, 100])
plt.plot(xpoints, ypoints)
plt.xlabel('Extrapolated depth')
plt.ylabel('Actual depth')

# distance plot
plt.figure()
plt.scatter(observed_dis, accurate_dis, color='b')
plt.plot(xpoints, ypoints)
plt.xlabel('Distance approximation')
plt.ylabel('Actual distance')

# new method 1
plt.figure()
plt.scatter(observed_dis1, accurate_dis1, color='g')
plt.plot(xpoints, ypoints)
plt.xlabel('Distance approximation 1')
plt.ylabel('Actual distance 1')

# new method 2
plt.figure()
plt.scatter(observed_dis2, accurate_dis2, color='g')
plt.plot(xpoints, ypoints)
plt.xlabel('Distance approximation 2')
plt.ylabel('Actual distance 2')

plt.show()
