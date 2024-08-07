# AI Virtual Canvas

AI Virtual Canvas is an innovative application that combines hand tracking, gesture-based painting, Optical Character Recognition (OCR), and more. This project leverages various technologies such as Python, TensorFlow, OpenCV, MediaPipe, and Pytesseract to create an interactive and intuitive virtual canvas.

**Note: This project is still in development.**

## Features

- **Real-Time Hand Tracking:** Uses MediaPipe for detecting and tracking hand movements in real-time.
- **Painter Using Hand Gesture:** Allows users to draw on the canvas using hand gestures.
- **OCR (Optical Character Recognition):** Recognizes and extracts text from hand-drawn sketches.
- **Clear Canvas:** Provides functionality to clear the canvas.
- **Layout Image with Color Change:** Users can change the color layout of the canvas.

## Tech Stack

- **Programming Language:** Python
- **Machine Learning:** TensorFlow
- **Computer Vision:** OpenCV, MediaPipe
- **OCR:** Pytesseract

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/EvanescenT07/AI-Virtual-Painter.git
   cd AI-Virtual-Painter
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   Ensure that Pytesseract is correctly installed and configured on your system.

4. **Run the application:**
   ```bash
   python AIVirtualPainter.py
   ```

## Usage

- **Real-Time Hand Tracking:** Move your hand in front of the webcam to see real-time tracking.
- **Painter Using Hand Gesture:** Use your index finger to draw on the canvas. The color and thickness of the drawing tool can be selected using the header.
- **OCR:** Click the OCR button to recognize and extract text from the canvas. The recognized text is saved in the `text/recognized_text.txt` file.
- **Clear Canvas:** Click the Clear button to clear the canvas.
