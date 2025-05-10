# Vehicle, People, and Traffic Sign Detection with Road Bend Analysis

This project is a Tkinter-based Windows application for real-time detection of vehicles, people, traffic signs, road boundaries, and road bends (curves, junctions, roundabouts) in driving videos. It uses the YOLOv8 model for object detection and OpenCV for road boundary and bend analysis, providing a user-friendly interface to visualize and interact with the detection results.

## Features
- **Video Panel**: Displays the selected driving video with a fixed height of 360 pixels, maintaining aspect ratio.
- **Object Detection**: Detects vehicles, people, and traffic signs using YOLOv8, showing bounding boxes, labels, confidence scores, and estimated distances.
- **Road Boundary Detection**: Identifies and draws road boundaries (red) and central lines (blue) using Canny edge detection and Hough Line Transform.
- **Bend Detection**: Detects upcoming road bends (left, right, junctions, roundabouts) based on line angles, with estimated distance to the bend.
- **GUI Controls**:
  - **Select Video**: Choose a video file (.mp4, .avi, .mov) for processing.
  - **Pause/Resume**: Pause or resume the video stream without clearing detection history.
  - **Stop Video**: Stop the video and clear the display.
  - **Confidence Threshold**: Adjust the YOLOv8 confidence threshold (0 to 1) for object detection.
  - **Slowdown Factor**: Adjust the video playback speed (1 = normal, >1 = slower).
- **Data Panel**: Displays detection results (object details, boundary lines, bend information) with persistent history.

## Requirements
- Python 3.8 or higher
- Required Python packages:
  ```bash
  pip install opencv-python ultralytics Pillow numpy tkinter
  ```
- YOLOv8 model weights (`yolov8n.pt`) will be automatically downloaded by the `ultralytics` package on first run.

## Installation
1. Clone or download this repository to your local machine.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the packages individually as listed in the Requirements section.
3. Ensure you have a compatible video file (.mp4, .avi, or .mov) for testing.

## Usage
1. Run the application:
   ```bash
   python vehicle_detection_app.py
   ```
2. The Tkinter GUI will open with the following components:
   - **Video Panel**: At the top, showing the processed video feed.
   - **Button Panel**: Contains "Select Video," "Pause/Resume," and "Stop Video" buttons.
   - **Data Panel**: Below the buttons, displaying detection results.
   - **Confidence Threshold Input**: Enter a value (0 to 1) to adjust object detection sensitivity.
   - **Slowdown Factor Input**: Enter a value (>=1) to slow down the video playback.
3. Click "Select Video" to choose a driving video. The application will start processing, displaying:
   - Detected objects (vehicles, people, traffic signs) with green bounding boxes and labels.
   - Road boundaries (red lines) and central lines (blue lines).
   - Bend information (e.g., "Approaching Left Bend, Distance: X units") when detected.
4. Use the "Pause/Resume" button to pause the video without losing detection data, or "Stop Video" to end the session.
5. Adjust the confidence threshold or slowdown factor as needed and click the respective "Update" or "Apply Slowdown" buttons.

## Project Structure
- `vehicle_detection_app.py`: Main application script containing the Tkinter GUI and detection logic.
- `README.md`: This file, providing project documentation.

## Notes
- **Bend Detection**: Bends are detected by analyzing the angles of road boundaries and central lines. The application is tuned to detect curves, T-junctions, and roundabouts, but performance may vary depending on video quality and road conditions.
- **Distance Estimation**: Distances for objects and bends are approximated using heuristics (bounding box size for objects, vertical line position for bends) and are not precise measurements.
- **Performance**: Processing speed depends on the system's hardware and the video resolution. Adjust the slowdown factor for smoother playback if needed.
- **Video Requirements**: Use clear, daytime driving videos with visible lane markings for best results.

## Limitations
- Bend detection may be less accurate in low-visibility conditions (e.g., night, fog) or with poorly marked roads.
- The application assumes a forward-facing camera with a standard perspective.
- Distance estimates are rough and should not be used for critical navigation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details (if applicable).

## Acknowledgments
- Built with [YOLOv8](https://github.com/ultralytics/ultralytics) for object detection.
- Uses [OpenCV](https://opencv.org/) for image processing and line detection.
- Tkinter for the GUI, provided by Python's standard library.
