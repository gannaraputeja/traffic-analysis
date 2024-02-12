import cv2
import pandas as pd
import time
from ultralytics import YOLO
from tracker import *

model = YOLO('yolov8s.pt')


def on_mouse_move(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coordinates = [x, y]
        print(mouse_coordinates)


windowName = 'Traffic Analysis'
cv2.namedWindow(windowName)
cv2.setMouseCallback(windowName, on_mouse_move)

cap = cv2.VideoCapture('videos/veh2.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

focus_objects = ["bicycle", "bus", "car", "motorcycle", "truck"]

count = 0
northbound_vehicles = dict()
southbound_vehicles = dict()

northbound_counter = list()
southbound_counter = list()

tracker = Tracker()

cy1 = 323
cy2 = 369
offset = 6

distance = 10  # meters

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    box_indices = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c in focus_objects:
            box_indices.append([x1, y1, x2, y2])
    bbox_id = tracker.update(box_indices)
    for bbox in bbox_id:
        x3, y3, x4, y4, box_id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        # mark object
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

        # southbound vehicles
        if (cy + offset) > cy1 > (cy - offset):
            southbound_vehicles[box_id] = time.time()
        if box_id in southbound_vehicles:
            if (cy + offset) > cy2 > (cy - offset):
                southbound_elapsed_time = time.time() - southbound_vehicles[box_id]
                southbound_speed_ms = distance / southbound_elapsed_time
                southbound_speed_kmh = southbound_speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(box_id), (x4, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"{southbound_speed_kmh:.2f} km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 255, 255), 1)
                if box_id not in southbound_counter:
                    southbound_counter.append(box_id)

        # northbound vehicles
        if (cy + offset) > cy2 > (cy - offset):
            northbound_vehicles[box_id] = time.time()
        if box_id in northbound_vehicles:
            if (cy + offset) > cy1 > (cy - offset):
                northbound_elapsed_time = time.time() - northbound_vehicles[box_id]
                northbound_speed_ms = distance / northbound_elapsed_time
                northbound_speed_kmh = northbound_speed_ms * 3.6
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(frame, str(box_id), (x4, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(frame, f"{northbound_speed_kmh:.2f} km/h", (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 255, 255), 1)
                if box_id not in northbound_counter:
                    northbound_counter.append(box_id)

    cv2.line(frame, (250, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(frame, 'line 1', (274, cy1 - offset), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    cv2.line(frame, (145, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(frame, 'line 2', (164, cy2 - offset), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    cv2.putText(frame, f"Northbound: {len(northbound_counter)}", (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                (255, 255, 255), 1)
    cv2.putText(frame, f"Southbound: {len(southbound_counter)}", (10, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                (255, 255, 255), 1)

    cv2.imshow(windowName, frame)
    if cv2.waitKey(0) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
