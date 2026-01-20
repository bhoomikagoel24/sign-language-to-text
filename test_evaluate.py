import cv2
import numpy as np
import math
import time
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D as KerasDepthwiseConv2D
from cvzone.HandTrackingModule import HandDetector
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from memory_profiler import profile

# === LOAD MODEL ===
class CustomDepthwiseConv2D(KerasDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        kwargs.pop("groups", None)
        super().__init__(*args, **kwargs)

model = load_model("Model/keras_model.h5", custom_objects={"DepthwiseConv2D": CustomDepthwiseConv2D})

labels = [chr(i) for i in range(65, 91)]  # A–Z

# === SETUP HAND DETECTOR ===
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 224

def preprocess_image(hand, img):
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    x, y, w, h = hand["bbox"]
    imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]

    if imgCrop.size == 0:
        return None

    aspectRatio = h / w
    if aspectRatio > 1:
        k = imgSize / h
        wCal = math.ceil(k * w)
        imgResize = cv2.resize(imgCrop, (wCal, imgSize))
        wGap = (imgSize - wCal) // 2
        imgWhite[:, wGap:wCal + wGap] = imgResize
    else:
        k = imgSize / w
        hCal = math.ceil(k * h)
        imgResize = cv2.resize(imgCrop, (imgSize, hCal))
        hGap = (imgSize - hCal) // 2
        imgWhite[hGap:hCal + hGap, :] = imgResize

    imgWhite = imgWhite.astype("float32") / 255.0
    return np.expand_dims(imgWhite, axis=0)

# === REAL-TIME PROCESSING ===
@profile
def process_frame(img):
    start = time.time()
    hands, img = detector.findHands(img)
    t1 = time.time()

    if hands:
        hand = hands[0]
        imgInput = preprocess_image(hand, img)

        if imgInput is not None:
            pstart = time.time()
            prediction = model.predict(imgInput)
            pend = time.time()

            label = labels[np.argmax(prediction)]

            cv2.putText(img, f"{label}", (hand["bbox"][0], hand["bbox"][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 255), 3)

            print(f"Detect: {(t1-start):.4f}s | Predict: {(pend-pstart):.4f}s")

    return img

# === MODEL EVALUATION ON TEST SET ===
def evaluate_test_set(test_dir="Test"):
    true_labels = []
    pred_labels = []

    for label in labels:
        folder = os.path.join(test_dir, label)
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.lower().endswith((".jpg", ".png")):
                img = cv2.imread(os.path.join(folder, file))
                hands, _ = detector.findHands(img)
                if not hands:
                    continue

                inp = preprocess_image(hands[0], img)
                if inp is None:
                    continue

                pred = model.predict(inp)
                pred_labels.append(labels[np.argmax(pred)])
                true_labels.append(label)

    if not true_labels:
        print("No test images found!")
        return

    acc = accuracy_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
    rec = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
    f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    print(f"\n=== METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

# === MAIN LOOP ===
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press ESC or Q to exit")

    while True:
        success, frame = cap.read()
        if not success:
            break

        out = process_frame(frame)
        cv2.imshow("Live Prediction", out)

        key = cv2.waitKey(1)
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Run evaluation after exiting
    evaluate_test_set("Test")
