"""
init.py
Used for recording data directory as well as project config
"""

cal_data_path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short_coords\\MTA_ext_short_coords\\train\\cam_{}"
bb_data_path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short\\MTA_ext_short\\train\\cam_{}"

sampling_rate = 5  # sampling rate in fps
BB_color = (0, 255, 0)  # bounding box color
dis_color = (0, 0, 255)  # distance line color
pid_color = (255, 255, 255)  # person id color

show_vid = True  # show video for distance detection

# path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short\\MTA_ext_short\\train\\cam_{}".format(cam_id)
# calibration_path = "C:\\Users\\Shadow\\Downloads\\MTA_ext_short_coords\\MTA_ext_short_coords\\train\\cam_{}".format(cam_id)