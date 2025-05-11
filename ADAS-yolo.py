import cv2
import numpy as np
import time
import threading
from picamera2 import Picamera2
from rpi_ws281x import *
from Motor import Motor
from Led import Led

# Initialize hardware
PWM = Motor()
led = Led()

# Global variables
enable_check_vehicle_position = True
lane_history_length = 10
left_lane_history = []
right_lane_history = []

# Distance Calculation Constants
REAL_HEIGHT_CM = 8.5
FOCAL_LENGTH = 276.47

# Load YOLO v4-tiny
net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Adaptive Canny edge detection for lane detection
def canny_edge(video_frame):
    gray = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return cv2.Canny(adaptive_thresh, 50, 150)

def calculate_distance(pixel_height):
    if pixel_height > 0:
        return (REAL_HEIGHT_CM * FOCAL_LENGTH) / pixel_height
    return None

# Object detection using YOLO
def detect_objects(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    objects_detected = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.6:
                center_x, center_y, w, h = (detection[:4] * [width, height, width, height]).astype("int")
                x, y = center_x - w // 2, center_y - h // 2
                label = classes[class_id]

                objects_detected.append((x, y, w, h, label))
                distance = calculate_distance(h)

                # LED response
                if label == "person":
                    led.colorWipe(led.strip, Color(255, 0, 0))
                elif label == "car":
                    led.colorWipe(led.strip, Color(255, 165, 0))
                else:
                    led.colorWipe(led.strip, Color(0, 0, 0))

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if distance:
                    cv2.putText(frame, f"{distance:.2f} cm", (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if len(objects_detected) == 0:
        led.colorWipe(led.strip, Color(0, 0, 0))  # Turn off LEDs

    return objects_detected, frame

# Lane helper functions
def filter_vertical_lines(lines, angle_threshold):
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if np.abs(angle - 90) < angle_threshold:
                vertical_lines.append(line)
    return vertical_lines

def smooth_lane_position(lane_history, new_value):
    lane_history.append(new_value)
    if len(lane_history) > lane_history_length:
        lane_history.pop(0)
    return int(np.mean(lane_history))

# Movement functions
def move_left():
    threading.Thread(target=lambda: (PWM.setMotorModel(-1400, -1400, 1400, 1400), time.sleep(0.5), PWM.setMotorModel(0, 0, 0, 0))).start()

def move_right():
    threading.Thread(target=lambda: (PWM.setMotorModel(1400, 1400, -1400, -1400), time.sleep(0.5), PWM.setMotorModel(0, 0, 0, 0))).start()

def move_forward():
    threading.Thread(target=lambda: (PWM.setMotorModel(700, 700, 700, 700), time.sleep(0.5), PWM.setMotorModel(0, 0, 0, 0))).start()

def stop_vehicle():
    PWM.setMotorModel(0, 0, 0, 0)

# Object avoidance based on position and distance
def check_vehicle_position(objects, left_lane, right_lane, width, height):
    image_center = width // 2
    relevant_objects = [(x, y, w, h, label, calculate_distance(h)) for (x, y, w, h, label) in objects if label in ["person", "car"]]

    for (x, y, w, h, label, distance) in relevant_objects:
        obj_center = x + w // 2
        if distance and distance < 25:
            print(f"⚠️ {label.capitalize()} detected at {distance:.2f} cm! Avoiding...")

            if obj_center < image_center:
                move_right(); time.sleep(1)
                move_forward(); time.sleep(2)
                move_forward(); time.sleep(2)
                move_left(); time.sleep(1)
                move_forward()
                return
            else:
                move_left(); time.sleep(1)
                move_forward(); time.sleep(2)
                move_forward(); time.sleep(2)
                move_right(); time.sleep(1)
                move_forward()
                return

    if left_lane <= image_center <= right_lane:
        move_forward()
    elif image_center < left_lane:
        move_right()
    else:
        move_left()

# Main program loop
if __name__ == "__main__":
    picam2 = Picamera2()
    width, height = 640, 480
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (width, height)}))
    picam2.start()

    left_lane = width // 4
    right_lane = 3 * width // 4

    while True:
        frame = picam2.capture_array()
        roi = frame.copy()

        # Lane detection
        canny_frame = canny_edge(roi)
        hough_lines = cv2.HoughLinesP(canny_frame, 1, np.pi / 180, 30, minLineLength=100, maxLineGap=40)
        vertical_lines = filter_vertical_lines(hough_lines, angle_threshold=60) if hough_lines is not None else []

        if vertical_lines:
            vertical_lines.sort(key=lambda x: x[0][0])
            left_lane = smooth_lane_position(left_lane_history, vertical_lines[0][0][0])
            right_lane = smooth_lane_position(right_lane_history, vertical_lines[-1][0][0])

        # Object detection
        objects_detected, frame = detect_objects(frame)

        # Obstacle avoidance and lane following
        if enable_check_vehicle_position:
            check_vehicle_position(objects_detected, left_lane, right_lane, width, height)

        # Display output
        resized_canny = cv2.resize(canny_frame, (320, 240))
        resized_frame = cv2.resize(frame, (320, 240))
        cv2.imshow('Lane Detection', resized_canny)
        cv2.imshow('Object Detection', resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
