# IPCV_project

## Description
The system captures live video using OpenCV and processes each frame in real time.  The Real-time face system can apply 3 Real-time Face Visual Effects, through keyboard interaction while the system is active.  This effects will be applied to the user’s face or/and hands.

## Requirements
- Python 3.8+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

### Keyboard commands
|Key|Command|Description|
|-|-|-|
|q / ESC|Quit|Stops the program and closes the window|
|0|Debug Mode|Shows the face and eyes bounding boxs|
|1|Big Eye Effect|Applies a warping effect to the eyes|
|2|Face Augmentation Effect|Places sunglasses on the deteted faces|
|3|Motion Tracking Effect|Detectes how your hand is positioned and when only the index finger is pointing a mustache is pasted on the user's face|
