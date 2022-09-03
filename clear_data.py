"""
clear_data.py
Used for clearing visually obstructed points in the input data file to improve distance detection accuracy
"""

import os
import time
import numpy as np
import pandas as pd
import init


# ================== helper fcn =======================================

def clear_noise(clear_num):
    """
    remove data of occluded people in new file, newfile in format "clear_{}_coords_cam_{}.csv".format(clear, cam_id)"
    clear_num: integer standing for number of occluded joints
    """

    # collect mask data into array
    x = 0
    print("Start processing data cleaning with clear={}...".format(clear_num))
    clear_indices = []
    process_time = time.time()
    while x < (file_length - joint_num):
        noise = annotated_csv.loc[x:(x + 21), ["joint_occluded"]].sum()
        noise = noise.iloc[0]
        if noise > clear_num:
            for y in range(joint_num):
                clear_indices.append(x + y)
        x += joint_num
        if x == joint_num * 1000:
            print("Finished 1k person data in %s seconds ---" % (time.time() - process_time))
            print("Predict finish all data in " + str(
                file_length / (joint_num * 1000) * (time.time() - process_time)) + " seconds ---")
    print("Finished noise data indices collection in %s seconds ---\n" % (time.time() - process_time))

    # new file
    new_df = annotated_csv[~annotated_csv.index.isin(clear_indices)]
    file_time = time.time()
    new_df.to_csv(os.path.join(init.BB_data_folder, "clear_{}_coords_cam_{}.csv".format(clear_complete, cam_id)))
    print("The file clear_{}_coords_cam_{}.csv".format(clear_complete, cam_id) + " is generated in %s seconds ---" % (
            time.time() - file_time))


# ================== main =============================================

def clear_complete(r1, r2):
    """this function clears the input dataframe and saves the cleared df in the range of [r1,r2]"""
    # multiple clear values for the cam_id
    for i in range(r1, r2 + 1):
        start_time = time.time()
        clear_noise(i)
        print("Total time for one run is %s seconds ---" % (time.time() - start_time))
        print("--------------------------------------------------------\n")
    print("\n--- ALL CLEARING FINISHED ---")


if __name__ == '__main__':
    # ================== initialization ===================================
    # testing params per camera
    # clear = number of total occluded joints allowed
    # generate cleared data files from the range [clear_r1, clear_r2]

    cam_id = 3  # MTA have 6 cams in total
    clear_r1 = 5
    clear_r2 = 11
    joint_num = 22  # a person have 22 joint identified

    # ================== reading data =====================================
    # read file
    clear = 0
    read_time = time.time()
    print("Reading file for cam_{}...".format(cam_id))

    cal_csv_path = os.path.join(init.BB_data_folder, "coords_cam_{}.csv".format(cam_id))
    annotated_csv = pd.read_csv(cal_csv_path)
    print("Finished reading coords_cam_{}.csv file in %s seconds ---".format(cam_id) % (time.time() - read_time))

    file_length = len(annotated_csv)
    print("File length is %s rows ---" % file_length)

    # clear empty cells in dataframe
    annotated_csv.replace('', np.nan, inplace=True)
    annotated_csv.dropna(inplace=True)
    print("Finished removing empty cells ---")
    print("--------------------------------------------------------\n")

    # ======================== clearing data ==============================
    clear_complete(clear_r1, clear_r2)
