# traffic analysis

- setup python environment and activate it [optional]

  ```bash
  python3.11 -m venv venv
  source venv/bin/activate
  ```

- install required dependencies

  ```bash
  pip3.11 install -r requirements.txt
  ```

- download `traffic_analysis.pt` and `traffic_analysis.mov` files

  ```bash
  ./setup.sh
  ```

## 🛠️ script arguments

- ultralytics

  - `--source_weights_path`: Required. Specifies the path to the YOLO model's weights
    file, which is essential for the object detection process. This file contains the
    data that the model uses to identify objects in the video.

  - `--source_video_path`: Required. The path to the source video file that will be
    analyzed. This is the input video on which traffic flow analysis will be performed.
  - `--target_video_path` (optional): The path to save the output video with
    annotations. If not specified, the processed video will be displayed in real-time
    without being saved.
  - `--confidence_threshold` (optional): Sets the confidence threshold for the YOLO
    model to filter detections. Default is `0.3`. This determines how confident the
    model should be to recognize an object in the video.
  - `--iou_threshold` (optional): Specifies the IOU (Intersection Over Union) threshold
    for the model. Default is 0.7. This value is used to manage object detection
    accuracy, particularly in distinguishing between different objects.

## ⚙️ run

- ultralytics

  ```bash
  python3.11 ultralytics_example.py \
  --source_weights_path data/traffic_analysis.pt \
  --source_video_path data/traffic_analysis.mov \
  --confidence_threshold 0.3 \
  --iou_threshold 0.5 \
  --target_video_path data/traffic_analysis_result.mov
  ```

