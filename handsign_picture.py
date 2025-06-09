import cv2
from ultralytics import YOLO

# Class ID to letter mapping (0 -> 'A', 1 -> 'B', ..., 25 -> 'Z')
label_to_char = {i: chr(65 + i) for i in range(26)}

# Load model
model = YOLO(r"C:\Users\use\OneDrive\Desktop\Sem 6\LAB\hand sign\best.pt")

# Load image
image_path = r"C:\Users\use\OneDrive\Desktop\Sem 6\LAB\hand sign\test\20240529_114507.jpg"  # <-- Update path here
image = cv2.imread(image_path)

# Resize function to fit within a window (preserves aspect ratio)
def resize_to_fit(img, max_width=800, max_height=800):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

if image is None:
    print(" Failed to load image. Please check the path.")
else:
    # Run detection
    results = model(image, conf=0.2)

    detected_labels = []
    image_copy = image.copy()

    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label_id = int(box.cls[0])
                
                if label_id in label_to_char:
                    char = label_to_char[label_id]
                    detected_labels.append(char)

                    cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 4)  
                    cv2.putText(image_copy, char, (x1, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)  

    # Build the sentence from detected letters
    sentence = "".join(detected_labels)
    print("\n Detected Letters:", detected_labels)
    print(" Formed Sentence:", sentence)

    # Resize and display the image
    resized_img = resize_to_fit(image_copy)
    cv2.imshow("Detection Result", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
