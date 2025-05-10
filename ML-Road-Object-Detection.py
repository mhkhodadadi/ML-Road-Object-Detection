import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os


class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vehicle, People, and Traffic Sign Detection with Road Bend Analysis")

        # Initialize YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.cap = None
        self.video_running = False
        self.is_paused = False
        self.confidence_threshold = 0.5
        self.slowdown_factor = 10  # Default: 1/10
        self.base_delay = 10  # Base delay in milliseconds

        # GUI Components
        # Video panel (fixed height)
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(pady=10)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()

        # Button panel
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)
        self.select_button = tk.Button(self.button_frame, text="Select Video", command=self.select_video)
        self.select_button.pack(side=tk.LEFT, padx=5)
        self.pause_button = tk.Button(self.button_frame, text="Pause", command=self.toggle_pause, state=tk.DISABLED)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(self.button_frame, text="Stop Video", command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Data panel
        self.data_text = scrolledtext.ScrolledText(self.root, width=60, height=6)
        self.data_text.pack(pady=10)

        # Confidence threshold input
        self.conf_frame = tk.Frame(self.root)
        self.conf_frame.pack(pady=5)
        tk.Label(self.conf_frame, text="Confidence Threshold:").pack(side=tk.LEFT)
        self.conf_entry = tk.Entry(self.conf_frame, width=10)
        self.conf_entry.insert(0, str(self.confidence_threshold))
        self.conf_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(self.conf_frame, text="Update", command=self.update_threshold).pack(side=tk.LEFT)

        # Slowdown factor input
        self.slowdown_frame = tk.Frame(self.root)
        self.slowdown_frame.pack(pady=5)
        tk.Label(self.slowdown_frame, text="Slowdown Factor (1 = normal, >1 = slower):").pack(side=tk.LEFT)
        self.slowdown_entry = tk.Entry(self.slowdown_frame, width=10)
        self.slowdown_entry.insert(0, str(self.slowdown_factor))
        self.slowdown_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(self.slowdown_frame, text="Apply Slowdown", command=self.update_slowdown).pack(side=tk.LEFT)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            self.stop_video()  # Stop any existing video
            self.cap = cv2.VideoCapture(file_path)
            if not self.cap.isOpened():
                self.data_text.delete('1.0', tk.END)
                self.data_text.insert(tk.END, "Error: Could not open video file.\n")
                return
            self.video_running = True
            self.is_paused = False
            self.pause_button.config(text="Pause", state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL)
            self.update_video()

    def stop_video(self):
        self.video_running = False
        self.is_paused = False
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_label.config(image='')
        self.data_text.delete('1.0', tk.END)

    def toggle_pause(self):
        if self.is_paused:
            self.is_paused = False
            self.video_running = True
            self.pause_button.config(text="Pause")
            self.update_video()
        else:
            self.is_paused = True
            self.video_running = False
            self.pause_button.config(text="Resume")

    def update_threshold(self):
        try:
            new_threshold = float(self.conf_entry.get())
            if 0 <= new_threshold <= 1:
                self.confidence_threshold = new_threshold
                self.data_text.delete('1.0', tk.END)
                self.data_text.insert(tk.END, f"Confidence threshold updated to: {self.confidence_threshold}\n")
            else:
                self.data_text.delete('1.0', tk.END)
                self.data_text.insert(tk.END, "Error: Threshold must be between 0 and 1.\n")
        except ValueError:
            self.data_text.delete('1.0', tk.END)
            self.data_text.insert(tk.END, "Error: Please enter a valid number.\n")

    def update_slowdown(self):
        try:
            new_factor = float(self.slowdown_entry.get())
            if new_factor >= 1:
                self.slowdown_factor = new_factor
                self.data_text.delete('1.0', tk.END)
                self.data_text.insert(tk.END, f"Slowdown factor updated to: {self.slowdown_factor}x\n")
            else:
                self.data_text.delete('1.0', tk.END)
                self.data_text.insert(tk.END, "Error: Slowdown factor must be >= 1.\n")
        except ValueError:
            self.data_text.delete('1.0', tk.END)
            self.data_text.insert(tk.END, "Error: Please enter a valid number.\n")

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        match_mask_color = 255
        cv2.fillPoly(mask, [vertices], match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)
        detected_text = ""

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    label = self.model.names[int(class_id)]
                    # Approximate distance (simplified, based on bounding box size)
                    bbox_width = x2 - x1
                    distance = max(1, 1000 / bbox_width)  # Simple heuristic
                    detected_text += f"Object: {label}, Confidence: {confidence:.2f}, Distance: {distance:.1f} units, BBox: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]\n"

                    # Draw bounding box and label
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
                    cv2.putText(frame, f"{label} {confidence:.1f}", (int(x1), int(y1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return frame, detected_text

    def update_video(self):
        if not self.video_running or not self.cap:
            return

        ret, frame = self.cap.read()
        if ret:
            # Resize frame to fixed height (360 pixels) while maintaining aspect ratio
            original_height, original_width = frame.shape[:2]
            fixed_height = 360
            aspect_ratio = original_width / original_height
            new_width = int(fixed_height * aspect_ratio)
            frame_resized = cv2.resize(frame, (new_width, fixed_height))

            # Process frame for object detection
            processed_frame, object_text = self.detect_objects(frame_resized)

            # Process frame for road boundary and bend detection
            processed_frame, boundary_text = self.detect_road_boundaries_and_curves(processed_frame)

            # Convert to ImageTk format
            img = Image.fromarray(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update detected objects and road info text
            self.data_text.delete('1.0', tk.END)
            combined_text = object_text or "No objects detected.\n"
            if boundary_text:
                combined_text += "\n" + boundary_text
            self.data_text.insert(tk.END, combined_text)

        else:
            self.stop_video()
            self.data_text.delete('1.0', tk.END)
            self.data_text.insert(tk.END, "Video ended or error occurred.\n")

        if self.video_running:
            # Adjust delay based on slowdown factor
            delay = int(self.base_delay * self.slowdown_factor)
            self.root.after(delay, self.update_video)

    def detect_road_boundaries_and_curves(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny edge detection with adjusted thresholds for better sensitivity
        edges = cv2.Canny(blur, 30, 120)

        # Define region of interest (trapezoid covering lower half of frame)
        height, width = frame.shape[:2]
        vertices = np.array([[
            (0, height),
            (width * 0.35, height * 0.55),
            (width * 0.65, height * 0.55),
            (width, height)
        ]], dtype=np.int32)

        # Apply ROI
        masked_edges = self.region_of_interest(edges, vertices)

        # Hough Line Transform with tuned parameters for complex roads
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=40, minLineLength=30, maxLineGap=100)

        boundary_text = ""
        left_lines = []
        right_lines = []
        central_lines = []
        min_y = height  # Track lowest y-coordinate for distance estimation

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Update minimum y for distance estimation
                min_y = min(min_y, y1, y2)

                # Calculate slope to classify lines
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1)
                    # Classify as left, right, or central based on slope and position
                    if slope < -0.3:  # Negative slope: left boundary
                        left_lines.append((slope, x1, y1, x2, y2))
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for boundaries
                        boundary_text += f"Left Boundary: [{x1}, {y1}] to [{x2}, {y2}]\n"
                    elif slope > 0.3:  # Positive slope: right boundary
                        right_lines.append((slope, x1, y1, x2, y2))
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red for boundaries
                        boundary_text += f"Right Boundary: [{x1}, {y1}] to [{x2}, {y2}]\n"
                    elif abs(slope) < 0.2 and (x1 + x2) / 2 > width * 0.4 and (x1 + x2) / 2 < width * 0.6:
                        # Near-vertical lines in the center: central line
                        central_lines.append((slope, x1, y1, x2, y2))
                        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)  # Blue for central line
                        boundary_text += f"Central Line: [{x1}, {y1}] to [{x2}, {y2}]\n"

        # Detect bends based on line angles and convergence
        bend_text = ""
        avg_left_slope = np.mean([s[0] for s in left_lines]) if left_lines else 0
        avg_right_slope = np.mean([s[0] for s in right_lines]) if right_lines else 0
        avg_central_slope = np.mean([s[0] for s in central_lines]) if central_lines else 0

        # Check for bends (curves, T-junctions, roundabouts)
        if (abs(avg_left_slope) > 0.5 or abs(avg_right_slope) > 0.5) and (left_lines or right_lines):
            # Estimate distance based on vertical position (heuristic: lower y = closer)
            distance = max(1, (height - min_y) * 0.1)  # Scale to arbitrary units
            # Analyze convergence and slopes for bend type
            if avg_left_slope < -0.5 and avg_right_slope > 0.5:
                # Converging lines (e.g., curve or roundabout)
                bend_text = f"Approaching Left Bend, Distance: {distance:.1f} units\n"
            elif avg_left_slope > -0.5 and avg_right_slope < 0.5:
                bend_text = f"Approaching Right Bend, Distance: {distance:.1f} units\n"
            elif abs(avg_central_slope) < 0.2 and central_lines:
                # Central line with diverging boundaries (e.g., T-junction or roundabout entry)
                if avg_left_slope > 0 or avg_right_slope < 0:
                    bend_text = f"Approaching Junction/Roundabout, Distance: {distance:.1f} units\n"

        # Only include bend information if a bend is detected
        boundary_text += bend_text
        return frame, boundary_text if boundary_text else ""


if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleDetectionApp(root)
    root.mainloop()
    