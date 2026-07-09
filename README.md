# Emotion Detector

## what it does

opens your webcam and detects your emotion in real time. slaps a bounding box on your face
and tells you if you look angry, happy, sad, disgusted, scared, surprised or just completely
neutral (me in every lecture)

7 emotions detected:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

## how it works

trained a CNN (convolutional neural network) on FER-2013, a dataset with like 35,000
labelled face images. the model learns what each emotion looks like pixel by pixel.
each frame gets converted to grayscale, a Haar cascade finds the face, the face gets
resized to 48x48 and normalized, then the CNN spits out a probability for each of the
7 emotions — the highest one wins and gets drawn on screen along with a bounding box
and a live bar chart of all 7 probabilities.

## stack

- Python
- TensorFlow / Keras
- OpenCV
- Gradio (browser-based version)
- NumPy

## Run locally

Two ways to run it — pick whichever you like:

- **`detect.py`** — opens a native desktop window using your webcam directly (OpenCV).
- **`app.py`** — same detection logic, but runs as a local web app in your browser (Gradio).

Prerequisites: Python 3.10–3.12 and a webcam.

**Windows (Command Prompt):**
```cmd
git clone https://github.com/sriman22-ui/Emotion-Detector.git
cd Emotion-Detector
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

Then run either:

**Desktop window** (native OpenCV window with your webcam feed):
```cmd
python detect.py
```

**Browser app** (opens `http://127.0.0.1:7860`):
```cmd
python app.py
```

**macOS/Linux:**
```bash
git clone https://github.com/sriman22-ui/Emotion-Detector.git
cd Emotion-Detector
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run either:

**Desktop window** (native OpenCV window with your webcam feed):
```bash
python detect.py
```

**Browser app** (opens `http://127.0.0.1:7860`):
```bash
python app.py
```

`detect.py`: press **Q** in the window to quit.
`app.py`: open the printed URL in your browser, click **"Click to Access Webcam"**, and
allow the permission prompt. Press **Ctrl+C** in the terminal to stop it.

the model file (`emotion_model.h5`) is included so you can skip training and just run
either script directly.

## Training

`train_model.py` trains the CNN from scratch on FER-2013 (`data/train`, `data/test`,
not included — download from Kaggle if you want to retrain). You don't need to run this
to use the app; the already-trained model is committed as `emotion_model.h5`.
