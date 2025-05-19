# Advanced Driver Assistance System Project

Abstract:
The aim of this project is to implement an Advanced Driver Assistance System (ADAS) capable of real-time object avoidance using integrated object detection and lane detection. By using a single camera on the front of the robot car, video input is captured, and it is processed using the lightweight YOLOv4-tiny model to identify objects and track the lane markings. The distance to the object is estimated using camera-based calculations, enabling the system to make appropriate navigation decisions to avoid collisions while maintaining the lane discipline. The prototype system, built on low-cost embedded hardware, demonstrated reliable performance in controlled environments using multithreaded control, efficient preprocessing techniques, and model optimization. This work highlights the feasibility of deploying ADAS features on resource-constrained platforms and contributes to the autonomous driving field by offering a compact, scalable, and efficient solution with clear potential for future enhancements.

The system combines:
- Object Detection using YOLOv4-tiny and SSD MobileNet
- Lane Detection with adaptive Canny edge detection
- Obstacle Avoidance based on detected object location and proximity
- Dynamic Lighting Adjustment for robust performance under varying conditions

Parameter Tuning:
To optimize detection accuracy and real-time responsiveness, both the lane detection and object detection systems were fine-tuned. For lane detection, the Canny edge detector and HoughLines were adjusted to reduce noise and ensure reliable vertical line detection, with dynamic thresholding and Gaussian blur improving robustness under varied lighting. The low_threshold value was interactively adjusted for optimal contrast between lane markings and the road background.Dynamic lighting adjustment was also incorporated for consistency in various lighting conditions.

For object detection, parameters such as confidence threshold and NMS threshold were calibrated to balance precision and avoid duplicate detections. The model input resolution was standardized with appropriate color conversion, and a light object detection model was selected for its speed and low resource requirements on Raspberry Pi.

Challenges Faced and Solutions Implemented:
Throughout the project development and implementation stages, several technical challenges were encountered across both the lane detection and object detection subsystems, as well as during integration with the Pi HQ camera. For lane detection, one major issue was the high sensitivity to lighting and noise, which led to false edge detection. This was resolved by applying Gaussian blur and morphological operations to smoothen the image, along with tuning the Hough Transform parameters to reduce irrelevant lines. Additionally, vehicle steering based on raw lane data caused jerky movements. This was mitigated by introducing threading and delay buffers to stabilize motor control. 

In object detection, LED signaling initially showed unreliable behavior due to timing and signal interference. A dedicated LED control class was implemented to manage clear signal transitions. Processing delays from running deep learning models on limited hardware were reduced by using the lightweight YOLOv4-tiny model. The camera also introduced latency during startup, which was improved by adding a brief sleep delay and discarding early frames. 

Using the Raspberry Pi HQ camera introduced additional challenges such as high latency due to large image resolutions, frame rate drops under heavy processing, and sequential delays from running object detection and lane detection in the same thread. These were resolved by reducing frame resolution, simplifying processing steps early in the pipeline, and introducing multithreading to run key functions in parallel. Further optimization included prioritizing obstacle detection when objects were within a critical range to ensure responsive system behavior.

Collectively, these troubleshooting efforts improved the real-time performance, stability, and reliability of the ADAS prototype, establishing a strong foundation for future enhancements in embedded intelligent vehicle systems.

Sustainability Note:
This project demonstrates that effective ADAS features can be deployed on low-cost, low-power hardware using lightweight models. Future enhancements may focus on optimizing model efficiency, refining obstacle avoidance logic, and adding sustainability-conscious features.

Acknowledgements
List of submodules utilized:
https://github.com/AlexeyAB/darknet
https://github.com/tensorflow/models
https://github.com/chuanqi305/MobileNet-SSD
http://host.robots.ox.ac.uk/pascal/VOC/