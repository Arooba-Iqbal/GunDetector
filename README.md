# 🔫 Real-Time Gun Detection with YOLOv8 and Python

This project is a real-time gun detection system built using **YOLOv8** (You Only Look Once, Version 8) and Python. It uses your webcam to detect firearms in a video stream, raises an alert through visual and audio cues, and displays the result in a simple but effective graphical interface.

### 📌 Project Description

The system detects guns in live video feed using a custom-trained YOLOv8 object detection model. When a gun is detected with confidence higher than **50%**, the application:
- Displays a red warning ("Gun Detected!") in the UI.
- Plays an **alarm sound** (`detection_alert.mp3`) as an additional alert.

The model was trained on a gun image dataset and fine-tuned to accurately identify firearms in various conditions. The application offers a simple GUI via `tkinter`, and runs on Windows.


### ⚙️ How It Works

1. **Model Loading:** The system loads a trained YOLOv8 model (`best_gun_yolo8.pt`).
2. **Webcam Feed:** It captures live video from your default webcam using OpenCV.
3. **Detection Loop:**
   - Each frame is passed through YOLO for detection.
   - If a gun is detected with over 50% confidence:
     - A red alert is shown.
     - A alarm sound are played.
4. **User Interface:** Uses `tkinter` to show the video feed and detection status in a GUI window.


### 🧰 Technologies Used

- **Python 3.10**
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- `opencv-python`
- `Pillow`
- `tkinter` (built-in GUI toolkit)
- `winsound` (built-in for Windows beep sounds)
- `.mp3` playback for alert


### 📂 File Structure

```
├── best_gun_yolo8.pt         # Trained YOLOv8 model
├── detection_alert.mp3       # Alert sound played on gun detection
├── gun3.py                   # Main Python script
├── ai-project.ipynb          # Training / model development notebook
├── .gitignore
├── README.md
```


### 🖥️ How to Run

#### 1. Clone the repository
```bash
git clone https://github.com/MuhammadWaleed-Animations/GunDetector.git
cd GunDetector
```

#### 2. Install dependencies
```bash
pip install opencv-python pillow ultralytics pygame winsound
```

> Make sure you’re using Python **3.10**. `tkinter` and `winsound` are built-in in standard Python for Windows.

#### 3. Run the application
```bash
python gun3.py
```

A GUI window will open showing the webcam feed. If a gun is detected with over 50% confidence, you will hear a sound and an alert will be shown.


### ⚠️ Notes

- **Audio Alerts:** The system uses `winsound` and `.mp3` playback for alerts. These work on **Windows** only.
- **Custom Model:** The `best_gun_yolo8.pt` file is a YOLOv8 model trained specifically on gun images. If you're training your own, make sure your dataset is well-labeled.
- **Performance:** The model runs inference in real time (20ms frame delay) but performance depends on your system's hardware.


### 🚀 Future Improvements

- Cross-platform sound alerts (for Linux/macOS)
- Logging and screenshot capture on detection
- Deployment as a desktop or web application


### 🙋‍♂️ Author

**Arooba Iqbal ** – Computer Science Student at FAST-NUCES
**Muhammad Waleed** – Computer Science Student at FAST-NUCES

