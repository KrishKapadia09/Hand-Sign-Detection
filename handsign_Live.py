import threading
import queue
import cv2
from ultralytics import YOLO
from collections import Counter
import time

label_counts = Counter()
counting_calls = 0
label_lock = threading.Lock()

detected_letters = []
last_detected_time = time.time()
letter_sentence = ""
last_letter = None
letter_lock = threading.Lock()

# Mapping class indices to letters
label_to_char = {i: chr(65 + i) for i in range(26)}  # 0 -> 'A', 1 -> 'B', ..., 25 -> 'Z'

# Adjustable parameters
DEBOUNCE_TIME = 1.0        # Seconds to ignore repeated detections
SPACE_THRESHOLD = 2.5      # Seconds of no detection = insert space

def capture_and_process(video, model, output_queue):
    global last_detected_time, last_letter, letter_sentence

    video = cv2.VideoCapture(0)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, persist=True, conf=0.2)
        res_plotted = results[0].plot()

        # Extract detections
        if results[0].boxes.id is not None:
            with label_lock:
                for det in results[0].boxes.data:
                    label_id = int(det[-1])
                    current_time = time.time()

                    # Convert to letter
                    if label_id in label_to_char:
                        char = label_to_char[label_id]

                        # Debounce logic
                        if char != last_letter or (current_time - last_detected_time) > DEBOUNCE_TIME:
                            with letter_lock:
                                # Add space if long gap since last detection
                                if (current_time - last_detected_time) > SPACE_THRESHOLD and letter_sentence and not letter_sentence.endswith(" "):
                                    letter_sentence += " "

                                letter_sentence += char
                                print(f"Added '{char}' -> {letter_sentence}")

                                last_letter = char
                                last_detected_time = current_time

        output_queue.put(res_plotted)

    video.release()
    output_queue.put(None)

def display_frames(output_queue):
    while True:
        frame = output_queue.get()
        if frame is None:
            break

        cv2.imshow("Tracking_Stream", frame)

        key = cv2.waitKey(10)
        if key == ord("x"):
            break

    cv2.destroyAllWindows()

    # Display final result
    print("\nFinal Detected Sentence:")
    print(letter_sentence.strip())

# Load model
model1 = YOLO(r"C:\Users\use\OneDrive\Desktop\Sem 6\LAB\hand sign\best.pt")

output_queue = queue.Queue(maxsize=10)

capture_thread = threading.Thread(target=capture_and_process, args=(0, model1, output_queue), daemon=True)
capture_thread.start()

display_frames(output_queue)
capture_thread.join()
