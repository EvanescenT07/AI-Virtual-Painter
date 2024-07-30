import cv2
import numpy as np
import os
import sys
import pytesseract
import HandTrackingModule as htm
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image, ImageOps
import tensorflow as tf 

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- Model and Data Paths ---
model_path = resource_path('model/bestModel.h5') # Path to your trained model
data_dir = 'Dataset/'  # Path to your Quick, Draw! dataset directory

# --- Load Your Trained Model ---
model = load_model(model_path)

# --- Get Class Names from Dataset Directory ---
class_names = sorted([filename[:-4] for filename in os.listdir(data_dir) if filename.endswith('.npy')])

# --- Drawing Settings ---
colorThickness = 10
eraserThickness = 50

# --- Load Overlay Images ---
imageLayout = "images"
appLayout = os.listdir(imageLayout)
overlay = []
for overlayList in appLayout:
    layout = cv2.imread(f"{imageLayout}/{overlayList}")
    overlay.append(layout)

# --- Initial Setup ---
header = overlay[0]  # Default header
drawColor = (0, 0, 255)  # Default drawing color (red)

# --- Webcam Setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

# --- Hand Detector ---
detector = htm.handDetector()

# --- Canvas and Variables ---
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
ocr_active = False
recognized_text = ""
model_active = False
detected_label = ""
is_drawing = False  # Flag to track if drawing is in progress

# --- Ensure text directory exists ---
if not os.path.exists('text'):
    os.makedirs('text')

# --- Preprocessing Functions ---
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return opening 

# --- Button Settings ---
buttonW, buttonH = 180, 50  
buttonSpacing = 50
buttonColor = (0, 0, 0) 
textColor = (255, 255, 255)
buttonRadius = 20

# --- Button Positions ---
headerHeight = 125
buttonX = 10
buttonModelY = headerHeight + buttonSpacing 
buttonOCR_Y = buttonModelY + buttonH + buttonSpacing
buttonClearY = buttonOCR_Y + buttonH + buttonSpacing

# --- Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        print("Error: Webcam not found.")
        break

    # --- Hand Detection and Drawing ---
    img = detector.findHands(img) 
    img = cv2.flip(img, 1)  
    lm, bbox = detector.findPosition(img, draw=False) 

    if len(lm) >= 9:
        for id in range(len(lm)):
            lm[id][1] = 1280 - lm[id][1] 

        x1, y1 = lm[8][1:] 
        x2, y2 = lm[12][1:] 
        fingers = detector.fingersUp() 

        if fingers[1] and fingers[2]: 
            xp, yp = 0, 0 
            is_drawing = False  # Not drawing while selecting
            if y1 < 125: # Header color selection
                if 335 < x1 < 450:  
                    header = overlay[0]
                    drawColor = (0, 0, 255)  
                elif 505 < x1 < 605: 
                    header = overlay[1]
                    drawColor = (0, 255, 0)  
                elif 675 < x1 < 775: 
                    header = overlay[2]
                    drawColor = (255, 0, 0)  
                elif 845 < x1 < 945: 
                    header = overlay[3]
                    drawColor = (255, 255, 255)  
                elif 1165 < x1 < 1265:
                    header = overlay[4]
                    drawColor = (0, 0, 0) 

            x_min, y_min = min(x1, x2), min(y1, y2)
            x_max, y_max = max(x1, x2), max(y1, y2)
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), drawColor, cv2.FILLED)

            # --- Button Click Handling ---
            if buttonX < x1 < buttonX + buttonW and buttonModelY < y1 < buttonModelY + buttonH:
                print("Model button clicked")
                model_active = True
                ocr_active = False

                # --- Find the bounding box of the drawing ---
                min_x, min_y = 1280, 720
                max_x, max_y = 0, 0
                for y in range(0, 720):
                    for x in range(0, 1280):
                        if imgCanvas[y, x, 0] != 0 or imgCanvas[y, x, 1] != 0 or imgCanvas[y, x, 2] != 0:
                            min_x = min(min_x, x)
                            max_x = max(max_x, x)
                            min_y = min(min_y, y)
                            max_y = max(max_y, y)

                # --- Extract and Preprocess Drawing ---
                current_drawing = imgCanvas[min_y:max_y, min_x:max_x]

                # --- Check if the drawing is empty ---
                if np.any(current_drawing):
                    current_drawing_gray = cv2.cvtColor(current_drawing, cv2.COLOR_BGR2GRAY)

                    # --- Flood Fill ---
                    h, w = current_drawing_gray.shape[:2]
                    seed_point = (w // 2, h // 2)
                    flood_color = 255
                    cv2.floodFill(current_drawing_gray, None, seed_point, flood_color)

                    # --- Create Mask ---
                    _, mask = cv2.threshold(current_drawing_gray, 200, 255, cv2.THRESH_BINARY)

                    # --- Apply Mask ---
                    masked_drawing = cv2.bitwise_and(current_drawing_gray, current_drawing_gray, mask=mask)

                    # --- Resize with Centering and Pad ---
                    resized_drawing = cv2.resize(masked_drawing, (28, 28), interpolation=cv2.INTER_AREA)
                    padded_drawing = np.zeros((28, 28), dtype=np.uint8)
                    x_offset = (28 - resized_drawing.shape[1]) // 2
                    y_offset = (28 - resized_drawing.shape[0]) // 2
                    padded_drawing[y_offset:y_offset+resized_drawing.shape[0], 
                                   x_offset:x_offset+resized_drawing.shape[1]] = resized_drawing

                    # --- Prepare for Model (Updated) ---
                    input_tensor = padded_drawing / 255.0  # Normalize directly
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    input_tensor = np.expand_dims(input_tensor, axis=-1)

                    # --- Make Prediction ---
                    prediction = model.predict(input_tensor)[0]
                    print("Confidence Scores:", prediction)

                    predicted_class_idx = np.argmax(prediction)
                    predicted_class_name = class_names[predicted_class_idx]

                    print(f"Predicted class: {predicted_class_name}")
                    detected_label = predicted_class_name
                else:
                    print("Drawing is empty. Please draw something.")

            elif buttonX < x1 < buttonX + buttonW and buttonOCR_Y < y1 < buttonOCR_Y + buttonH: 
                print("OCR button clicked")
                preprocessed = preprocess_image(imgCanvas)
                recognized_text = pytesseract.image_to_string(
                    preprocessed, config='--psm 6')
                print("Recognized Text:", recognized_text)
                with open("text/recognized_text.txt", "w") as text_file:
                    text_file.write(recognized_text)
                ocr_active = True
                model_active = False

            elif buttonX < x1 < buttonX + buttonW and buttonClearY < y1 < buttonClearY + buttonH:
                print("Clear button clicked")
                imgCanvas = np.zeros((720, 1280, 3), np.uint8)
                recognized_text = ""
                detected_label = ""
                ocr_active = False
                model_active = False

        # --- Drawing on Canvas ---
        if fingers[1] and not fingers[2]: 
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1  
            is_drawing = True  # Drawing in progress

            if drawColor == (0, 0, 0): # Eraser
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, colorThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, colorThickness)
            xp, yp = x1, y1 

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    # --- Overlay Header and Buttons ---
    img[0:125, 0:1280] = header

    cv2.rectangle(img, (buttonX, buttonModelY), (buttonX + buttonW, buttonModelY + buttonH), buttonColor, cv2.FILLED)
    cv2.putText(img, "Model", (buttonX + 40, buttonModelY + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)
    cv2.rectangle(img, (buttonX, buttonOCR_Y), (buttonX + buttonW, buttonOCR_Y + buttonH), buttonColor, cv2.FILLED)
    cv2.putText(img, "OCR", (buttonX + 60, buttonOCR_Y + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)
    cv2.rectangle(img, (buttonX, buttonClearY), (buttonX + buttonW, buttonClearY + buttonH), buttonColor, cv2.FILLED)
    cv2.putText(img, "Clear", (buttonX + 45, buttonClearY + 35), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)

    # --- Display Text if OCR Active ---
    if ocr_active:
        cv2.putText(img, f"OCR: {recognized_text}", (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)
    elif model_active:
        cv2.putText(img, f"Detected: {detected_label}", (50, 680), cv2.FONT_HERSHEY_SIMPLEX, 1, textColor, 2)

    cv2.imshow("AI Virtual Painter", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
