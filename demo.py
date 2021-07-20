import time
from pathlib import Path
import depthai as dai
import cv2
import numpy as np
from inference_helper import InferenceHelper

parent_dir = Path(__file__).parent
nn_path = str(
    (parent_dir / Path("models/mobilenet-ssd_openvino_2021.2_8shave.blob"))
    .resolve()
    .absolute()
)
video_path = str((parent_dir / Path("models/construction_vest.mp4")).resolve().absolute())

labelMap = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

def preprocess_func(frame):
    frame = cv2.resize(frame, (800, 450))
    return frame

def postprocess_func(raw_nn_results):
    nn_results = []
    detections = raw_nn_results.detections
    for detection in detections:
        nn_results.append(
            [
                detection.xmin,
                detection.ymin,
                detection.xmax,
                detection.ymax,
                detection.label,
                detection.confidence,
            ]
        )
    return nn_results

def display_func(frame, nn_results):
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    for result in nn_results:
        bbox = frameNorm(frame, (result[0], result[1], result[2], result[3]))
        cv2.putText(
            frame,
            labelMap[result[4]],
            (bbox[0] + 10, bbox[1] + 20),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.putText(
            frame,
            f"{int(result[5] * 100)}%",
            (bbox[0] + 10, bbox[1] + 40),
            cv2.FONT_HERSHEY_TRIPLEX,
            0.5,
            255,
        )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.imshow("Result", frame)

def wait_process_done(f, wait_time=0.001):
    # Monitor the status of another process 
    if not f.is_alive():
        time.sleep(0.001)
    print('display process is done.')

if __name__ == '__main__':
    my_inference_helper = InferenceHelper(
        video_path,
        preprocess_func,
        nn_path,
        (300, 300),
        2,
        postprocess_func,
        display_func,
        lock_fps=27,
        is_mobilenetdetection_nn=True,
    )
    my_inference_helper.run()
