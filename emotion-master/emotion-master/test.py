from pathlib import Path
import numpy as np
import argparse
import time
import os
import time
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
import math
import psycopg2

import torch.backends.cudnn as cudnn
import torch
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    scale_coords, set_logging, create_folder
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def detect(opt):
    source, view_img, imgsz, nosave, show_conf, save_path, show_fps = opt.source, not opt.hide_img, opt.img_size, opt.no_save, not opt.hide_conf, opt.output_path, opt.show_fps
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    # Directories
    create_folder(save_path)


    # Load model
    model = attempt_load(model1, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    Image.init(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

# Initialize YOLO model
model1 = YOLO('latest129.pt')

# Define class names
className = ['Ahmedalla Hani', 'Aisya Azhar', 'Hui Shan', 'Izlin Syamira', 'Kim Teck Lim', 'Mardhiah Nasri']

# Database connection parameters
conn_params = {
    "host": "localhost",
    "user": "postgres",
    "password": "dede7dede",
    "database": "attendAI"
}

# Initialize a flag to keep track of the last detection time

last_detection_time = None
message_display_duration = 4
display_message = False

# Function to retrieve employee data from PostgreSQL based on class name
def get_employee_data(class_name, conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM employee WHERE name = %s;", (class_name,))
    result = cursor.fetchone()
    cursor.close()
    return result  # Assuming the data is returned as a tuple or None if not found

# Function to update clock-in time for an employee
# def update_clock_in_time(employeeID, conn):
#     cursor = conn.cursor()
#     current_time = datetime.now()
#     cursor.execute("INSERT INTO employee_attendance (employeeID, clock_in) VALUES (%s, %s);", (employeeID, current_time))
#     conn.commit()
#     cursor.close()

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1920)  
cap.set(4, 1080)  

img_counter = 0

while True:
    success, img = cap.read()
    results = model1(img, stream=True)
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



    # Coordinates
    for r in results:
        boxes = r.boxes

        if len(boxes) == 0:
            # No detections
            org = (10, 50)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (0, 0, 0)  # Red color
            thickness = 2
            cv2.putText(img, "No face detected", org, font, fontScale, color, thickness)
        else:
            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convert to int values

                # Put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                if confidence > 0.8:
                    print("Confidence --->", confidence)

                    # Class name
                    class_idx = int(box.cls[0])
                    class_name = className[class_idx]
                    print("Class name -->", class_name)

                    # Retrieve employee data from PostgreSQL
                    conn = psycopg2.connect(**conn_params)
                    employee_data = get_employee_data(class_name, conn)
                    conn.close()

                    if employee_data:
                        # Employee data found
                        employee_id, name, age, position = employee_data
                        details = f"Name: {name}\nID: {employee_id}\nAge: {age}\nPosition: {position}"
                        # Update the last detection time when an employee is detected
                        if last_detection_time is None:
                            last_detection_time = datetime.now()
                            display_message = True
                    else:
                        details = "Unknown"

                    # Object details
                    org1 = [x1, y1 - 10]
                    org2 = [x1, y1 - 30]  # Adjust the vertical position to place it below the confidence
                    org3 = [x1, y1 - 50]  # Adjust the vertical position to place it below the first line
                    org4 = [x1, y1 - 70]  # Adjust the vertical position to place it below the second line
                    org5 = [x1, y1 - 90]  # Adjust the vertical position to place it below the second line

                    font = cv2.FONT_HERSHEY_TRIPLEX
                    fontScale = 0.5
                    color = (255, 0, 0)
                    thickness = 1

                    # Display confidence
                    cv2.putText(img, f"Confidence: {str(confidence)}", org4, font, fontScale, color, thickness)
                    # Display details below confidence, each detail on a new line
                    cv2.putText(img, "ID: " + str(employee_id), org3, font, fontScale, color, thickness)
                    cv2.putText(img, "Name: " + str(name), org2, font, fontScale, color, thickness)
                    # cv2.putText(img, "Age: " + str(age), org2, font, fontScale, color, thickness)
                    cv2.putText(img, "Position: " + position, org1, font, fontScale, color, thickness)

                else:
                    org = [x1, y1 - 10]
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    fontScale = 1
                    color = (0, 0, 255)  # Red color
                    thickness = 1
                    cv2.putText(img, "Unknown", org, font, fontScale, color, thickness)

        # Check if it's time to display the message
    if display_message:
        message = "Name: {}\nID: {}\nClockedIn: {}".format(name, employee_id, last_detection_time.strftime("%Y-%m-%d %H:%M"))
        org_message = (img.shape[1] - 200, 30)
        font_message = cv2.FONT_HERSHEY_SIMPLEX
        fontScale_message = 0.4
        color_message = (0, 0, 255)  # Red color
        thickness_message = 1

        

        # Split the message by newline characters and display each line separately
        lines = message.split('\n')
        for i, line in enumerate(lines):
            line_y = org_message[1] + i * 20  # Adjust the vertical position for each line
            cv2.putText(img, line, (org_message[0], line_y), font_message, fontScale_message, color_message, thickness_message)

        # Check if the message display duration has passed
        if (datetime.now() - last_detection_time).total_seconds() >= message_display_duration:
            last_detection_time = None
            display_message = False
    
    cv2.putText(img, current_datetime, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # Save the frame with the detected faces as an image
    image = Image.fromarray(img)
    image.save('output/image_{}.jpg'.format(img_counter))

    # Increment the frame counter
    img_counter += 1

    cv2.namedWindow('AI Attendance', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AI Attendance', 1280, 720) 
    cv2.imshow('AI Attendance', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='face confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--hide-img', action='store_true', help='hide results')
    save = parser.add_mutually_exclusive_group()
    save.add_argument('--output-path', default="output.mp4", help='save location')
    save.add_argument('--no-save', action='store_true', help='do not save images/videos')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--show-fps', default=False, action='store_true', help='print fps to console')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))
    with torch.no_grad():
        detect(opt=opt)
