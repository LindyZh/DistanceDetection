import os
import time
import numpy as np
import pandas as pd


# ================== initialization ===================================
# testing params
# clear = number of total occluded joints allowed
# generate clear from clear_r1 to clear_r2 (include clear_r2)
cam_id = 3
clear_r1 = 5
clear_r2 = 11


# ================== reading data =====================================
#read file
clear = 0
read_time = time.time()
print("Reading file for cam_{}...".format(cam_id))
calibration_path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short_coords\\MTA_ext_short_coords\\train\\cam_{}".format(cam_id)
cal_csv_path = os.path.join(calibration_path, "coords_cam_{}.csv".format(cam_id))
annotated_csv = pd.read_csv(cal_csv_path)
print("Finished reading coords_cam_{}.csv file in %s seconds ---".format(cam_id) % (time.time() - read_time))
file_length = len(annotated_csv)
print("File length is %s rows ---" % file_length)

#clear empty cells in dataframe
annotated_csv.replace('', np.nan, inplace=True)
annotated_csv.dropna(inplace=True)
print("Finished empty cells removing ---" )
print("--------------------------------------------------------\n")


# ================== helper fcn =======================================
#remove data of occluded people in new file
#new file in format "clear_{}_coords_cam_{}.csv".format(clear, cam_id)"
#noise = number of occluded joints
def clear_noise(clear):
    #collect mask data into array
    x = 0
    print("Start processing data cleaning with clear={}...".format(clear))
    new_df = pd.DataFrame()
    clear_indices = []
    process_time = time.time()
    while(x < (file_length-22)):
        noise = 0
        noise = annotated_csv.loc[x:(x+21),["joint_occluded"]].sum()
        noise = noise.iloc[0]
        if(noise > clear):
            for y in range(22):
                clear_indices.append(x+y)
        x += 22
        if(x == 22000):
            print("Finished 22k data in %s seconds ---" % (time.time() - process_time))
            print("Predict finish all data in " + str(file_length/22000*(time.time()-process_time)) + " seconds ---")
    print("Finished noise data indices collection in %s seconds ---\n" % (time.time() - process_time))

    #new file
    new_df = annotated_csv[~annotated_csv.index.isin(clear_indices)]
    file_time = time.time()
    new_df.to_csv(os.path.join(calibration_path,"clear_{}_coords_cam_{}.csv".format(clear, cam_id)))
    print("The file clear_{}_coords_cam_{}.csv".format(clear, cam_id) + " is generated in %s seconds ---" % (time.time() - file_time))


# ================== main =============================================
#multiple clear values for the cam_id
for i in range(clear_r1, clear_r2+1):
    start_time = time.time()
    clear = i
    clear_noise(clear)
    print("Total time for one run is %s seconds ---" % (time.time() - start_time))
    print("--------------------------------------------------------\n")
print("\n--- ALL FINISHED ---")
