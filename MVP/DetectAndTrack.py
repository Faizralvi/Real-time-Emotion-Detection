from ultralytics import YOLO
import subprocess
import cv2
import numpy as np
import argparse
from collections import defaultdict
from threading import Thread
import time
import os

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = Thread(target=self.update, args=()).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()
    
    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

class FPS:
    def __init__(self):
        self.prev_time = time.time()
        self.curr_time = 0
        self.frame_count = 0
        self.fps = 0
        self.update_interval = 1

    def update(self):
        self.frame_count += 1
        self.curr_time = time.time()
        time_diff = self.curr_time - self.prev_time
        
        if time_diff >= self.update_interval:
            self.fps = self.frame_count / time_diff
            self.frame_count = 0
            self.prev_time = self.curr_time
        
        return self.fps

def get_video_stream_url(youtube_url):
    formats = {
        "270": "1080",
        "311": "720p60",
        "232": "720p",
        "136": "720p",
        "135": "480p",
        "134": "360p",
        "133": "240p",
        "160": "144p"
    }
    for fmt in formats.keys():
        cmd = ["yt-dlp", "-g", "-f", fmt, youtube_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_url = result.stdout.strip()
        if video_url:
            print(f"\r✅ Found video URL for {formats[fmt]}")
            return video_url
        else:
            print(f"\r❌ Format {formats[fmt]} not available, trying lower quality...")
    print("\r❌ No valid MP4 video format found.")
    return None

def main():
    parser = argparse.ArgumentParser(description='YOLO Object Tracking with YouTube or Webcam')
    parser.add_argument('--source', type=str, default='webcam', help='Source type: "webcam" or "youtube"')
    parser.add_argument('--youtube_url', type=str, default='https://youtu.be/su33E1lreMc?si=b2ritLiv6uCMKOx3', help='YouTube URL if source is youtube')
    parser.add_argument('--webcam_id', type=int, default=0, help='Webcam device ID if source is webcam')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model')
    parser.add_argument('--conf', type=float, default=0.4, help='Confidence threshold')
    args = parser.parse_args()

    print(f"Loading YOLO model: {args.model}")
    if os.path.exists(args.model):
        model_path = args.model
    else:
        model_path = os.path.join(current_dir, 'models', args.model)
        
    model = YOLO(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    if args.source == 'youtube':
        stream_url = get_video_stream_url(args.youtube_url)
        if not stream_url:
            print("Failed to get YouTube video URL. Exiting.")
            return
        vs = VideoStream(stream_url)
    else:
        vs = VideoStream(args.webcam_id)

    fps_counter = FPS()
    track_history = defaultdict(lambda: [])
    frame_counter = 0
    running = True

    while running:
        ret, frame = vs.read()
        if not ret or frame is None:
            print("\n⚠️ Stream ended or failed.")
            break

        result = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=args.conf,
            iou=0.5,
            imgsz=640,
            stream=False
        )[0]

        draw_frame = frame.copy()
        total_person_detected = 0

        if result.boxes and hasattr(result.boxes, 'id') and result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            total_person_detected = len(track_ids)

            for box, track_id in zip(boxes, track_ids):
                x_center, y_center, w, h = box
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(draw_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                track = track_history[track_id]
                track.append((float(x_center), float(y_center)))
                if len(track) > 30:
                    track.pop(0)

                if len(track) > 1:
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(draw_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                if frame_counter % 1 == 0:
                    y1_safe = max(0, y1)
                    y2_safe = min(frame.shape[0], y2)
                    x1_safe = max(0, x1)
                    x2_safe = min(frame.shape[1], x2)

                    if y2_safe > y1_safe and x2_safe > x1_safe:
                        person_roi = frame[y1_safe:y2_safe, x1_safe:x2_safe]
                        if person_roi.size > 0 and person_roi.shape[0] > 20 and person_roi.shape[1] > 20:
                            try:
                                gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
                                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20), maxSize=(300, 300))
                                for (fx, fy, fw, fh) in faces:
                                    abs_x1 = fx + x1_safe
                                    abs_y1 = fy + y1_safe
                                    abs_x2 = fx + fw + x1_safe
                                    abs_y2 = fy + fh + y1_safe
                                    cv2.rectangle(draw_frame, (abs_x1, abs_y1), (abs_x2, abs_y2), (0, 255, 0), 2)
                                    cv2.putText(draw_frame, "Face", (abs_x1, abs_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error processing face detection: {e}")

        current_fps = fps_counter.update()
        source_text = f"Source: {'YouTube' if args.source == 'youtube' else 'Webcam'}"
        fps_text = f"FPS: {current_fps:.1f}"

        cv2.putText(draw_frame, source_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(draw_frame, fps_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(draw_frame, f"Total Persons: {total_person_detected}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow('Detect and Track', draw_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            running = False

        frame_counter += 1

    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()