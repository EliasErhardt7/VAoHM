# VAoHM
##Modules:
saveFeatures.py → feature extraction CLI.
trainClassifier.py → model training.
poseDetection.py → live / offline inference, GUI overlay.
performance_evaluation.py → DTW scoring utilities.
##Dependencies: To run the program the following packages need to be installed:
Python 3.10, scikit-learn 1.5, fastdtw 0.3.0, OpenCV 4.10, Pandas 2.2.3, MediaPipe 0.10.21.
I/O: Video paths or webcam index via CLI flags; scoring & FPS logged to CSV for later analysis.
