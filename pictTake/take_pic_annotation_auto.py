import cv2 
import csv
import time
import os
import numpy as np
img_counter = 0  
vid         = cv2.VideoCapture(0) 

label       = "rubber_trad"
total_pic   = 3
interval_pic= 0.15
mode        = "test"
name_prefix = mode+label+"_"
folder_name = mode+"/"
start_name  = 2008
end_name    = start_name+total_pic
take_flag   = 0
SEGMENT     = 1
# captured area
ymin_cap    = 0
ymax_cap    = 416
xmin_cap    = 0
xmax_cap    = 416

# new labeled area
Sx                  = {}
Sy                  = {}
color_scale         = (0,255,255)
BOX_RAD             = 55 
radius_holder       = 5 

# SEGMENT A1
Sx[1]               = 170
Sy[1]               = 180

# SEGMENT A3
Sx[3]               = 320
Sy[3]               = 180

# SEGMENT A5
Sx[5]               = 470
Sy[5]               = 180

# SEGMENT B2
Sx[2]               = 170
Sy[2]               = 310


# SEGMENT B4
Sx[4]               = 320
Sy[4]               = 310

# SEGMENT B6
Sx[6]               = 470
Sy[6]               = 310

x_rad_min     = Sx[SEGMENT]-BOX_RAD
x_rad_max     = Sx[SEGMENT]+BOX_RAD
y_rad_min     = Sy[SEGMENT]-BOX_RAD
y_rad_max     = Sy[SEGMENT]+BOX_RAD
SEGMENT_COOR  = {'x_rad_min' :x_rad_min, 'x_rad_max'   : x_rad_max, 'y_rad_min'  :y_rad_min,     'y_rad_max' : y_rad_max}

def draw_bracket(image_np, color_scale, bracket_coordinates, x, y):
    cv2.circle(image_np, (x,y), radius_holder, (0, 0, 255), 4)
    pts_guide = np.array([ [bracket_coordinates['x_rad_min'],bracket_coordinates['y_rad_min'] ]
                         , [bracket_coordinates['x_rad_max'],bracket_coordinates['y_rad_min'] ]
                         , [bracket_coordinates['x_rad_max'],bracket_coordinates['y_rad_max'] ]
                         , [bracket_coordinates['x_rad_min'],bracket_coordinates['y_rad_max'] ] 
                         ], np.int32)
    pts_guide = pts_guide.reshape((-1,1,2))
    cv2.polylines(image_np,[pts_guide],True,color_scale,2)    
if not os.path.exists(folder_name):
    os.mkdir(folder_name)

print("press space when ready")
while(True): 
    ret, frame              = vid.read()
    frame_ori               = 0 
    frame_resize            = frame[ymin_cap:ymax_cap, xmin_cap:xmax_cap]
    frame_label             = frame[y_rad_min:y_rad_max, x_rad_min:x_rad_max]
    frame_resize_label      = frame_resize
    cv2.imshow('frame_original',        frame) 

    key_press = cv2.waitKey(1)
    if key_press%256 == 32:
        take_flag  =1
    if img_counter == total_pic:
        print(str(total_pic)+" Finish")
        break
    elif take_flag==1:
        name        = start_name+img_counter
        img_name    = name_prefix+str(name)+".jpg".format(img_counter)
        cv2.imwrite(folder_name+img_name, frame_resize)
        print("{} written!".format(img_name))
        img_counter += 1
    # draw bracket for adjustment only
    draw_bracket(frame_resize_label, color_scale, SEGMENT_COOR, Sx[SEGMENT], Sy[SEGMENT])
    cv2.imshow('frame_resize_label_write',    frame_resize_label)
    cv2.imshow('frame_label', frame_label)
    cv2.imshow('frame_original_BRACKET',        frame)
    time.sleep(interval_pic)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

def xml_to_csv_train(total_pic):
    with open(folder_name+'train.csv', 'w', newline='') as csvfile:
        column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
        writer = csv.DictWriter(csvfile, fieldnames=column_name)
        writer.writeheader()
        for x in range(total_pic):
            name = start_name+x
            writer.writerow({'filename': name_prefix+str(name)+".jpg", 'width': xmax_cap, 'height': ymax_cap
                            , 'class'  : label
                            , 'xmin'   : x_rad_min
                            , 'ymin'   : y_rad_min
                            , 'xmax'   : x_rad_max
                            , 'ymax'   : y_rad_max})
        print("csv created")
# After the loop release the cap object 
xml_to_csv_train(total_pic)
vid.release() 
cv2.destroyAllWindows() 