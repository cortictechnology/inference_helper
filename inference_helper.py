import cv2
import depthai as dai
import numpy as np
import time
import collections
import threading
import multiprocessing as mp

# Notes:
# 1. Only support local video file as input_src currently, but you can modify it easily to support OAK-D camera input.
# 2. Only support the packing of 1 NN model, but it is also very doable to modify the code to support the packing of more
#    NN models.

def display_frame(
    display_func,
    frame_queue,
    nn_queue,
    nn_result_ready_flag,
    num_parallel_frames,
    lock_fps,
):
    fps = 0.0
    FRAME_DISPLAY_TIME = 1.0 / lock_fps
    while True:
        if bool(nn_result_ready_flag.value):
            start_time = time.perf_counter()
            nn_result_ready_flag.value = 0
            for i in range(num_parallel_frames):
                t0 = time.perf_counter()
                frame = frame_queue.get()
                nn_results = nn_queue.get()
                cv2.putText(
                    frame,
                    "FPS: " + str(fps),
                    (20, 20),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    255,
                )
                display_func(frame, nn_results)
                time_used = time.perf_counter() - t0
                time_left = FRAME_DISPLAY_TIME - time_used
                if time_left > 0:
                    sleep_time = time_left - 0.0006
                    cv_wait_time = int(sleep_time * 1000)
                    if cv_wait_time <= 0:
                        cv_wait_time = 1
                    cv2.waitKey(cv_wait_time)
            fps = round(1 / ((time.perf_counter() - start_time) / num_parallel_frames), 1)
            print("Drawing time:", time.perf_counter() - start_time)
        else:
            time.sleep(0.001)


class InferenceHelper:
    def __init__(
        self,
        video_location,
        preprocess_func,
        nn_model_path,
        nn_input_size,
        num_parallel_frames,
        postprocess_func,
        display_func,
        lock_fps=20,
        is_mobilenetdetection_nn=False,
    ):
        self.manager = mp.Manager()
        self.frame_queue = self.manager.Queue()
        self.nn_queue = self.manager.Queue()
        self.grab_frame_flag = True
        self.frame_ready_flag = False
        self.nn_result_ready_flag = mp.Value("i", 0)
        self.num_parallel_frames = num_parallel_frames
        self.nn_input_size = nn_input_size
        self.lock_fps = lock_fps
        self.frame_lock_time = 1.0 / lock_fps

        self.frame_event = threading.Event()

        # you can modify this to support OAK-D camera or webcam input
        self.input_src = cv2.VideoCapture(video_location)

        # preprocess_func should only take in a cv frame as its sole argument
        self.preprocess_func = preprocess_func

        # postprocess_funcshould only take in a depthai.ADatatype as its sole argument
        self.postprocess_func = postprocess_func

        # display_func should only take in a cv frame and a data object as its arguments
        self.display_func = display_func

        pipeline = dai.Pipeline()
        # Uncomment this line for model compatibility
        # pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_2)

        if is_mobilenetdetection_nn:
            nn = pipeline.createMobileNetDetectionNetwork()
            nn.setConfidenceThreshold(0.5)
        else:
            nn = pipeline.createNeuralNetwork()
        nn.setBlobPath(nn_model_path)
        nn.input.setBlocking(False)

        nn_in = pipeline.createXLinkIn()
        nn_out = pipeline.createXLinkOut()

        nn_in.setStreamName("nn_in")
        nn_out.setStreamName("nn_out")

        nn_in.out.link(nn.input)
        nn.out.link(nn_out.input)

        self.device = dai.Device(pipeline)
        self.nn_in_queue = self.device.getInputQueue(name="nn_in")
        self.nn_out_queue = self.device.getOutputQueue(
            name="nn_out", maxSize=self.num_parallel_frames, blocking=False
        )

        self.grab_frame_thread = threading.Thread(target=self.grab_frame, daemon=True)
        self.process_frame_thread = threading.Thread(
            target=self.process_frame, daemon=True
        )
        self.display_process = mp.Process(
            target=display_frame,
            args=(
                self.display_func,
                self.frame_queue,
                self.nn_queue,
                self.nn_result_ready_flag,
                self.num_parallel_frames,
                self.lock_fps,
            ),
        )

    def to_planar(self, arr: np.ndarray, shape: tuple):
        return (
            cv2.resize(arr, shape, interpolation=cv2.INTER_NEAREST)
            .transpose(2, 0, 1)
            .flatten()
        )

    def generate_nn_frame(self, frame):
        img = dai.ImgFrame()
        image = self.to_planar(frame, (300, 300))
        img.setData(image)
        img.setTimestamp(time.monotonic())
        img.setWidth(self.nn_input_size[0])
        img.setHeight(self.nn_input_size[1])
        return img

    def grab_frame(self):
        while True:
            if self.grab_frame_flag:
                nn_imgs = collections.deque(maxlen=self.num_parallel_frames)
                counter = 0
                while counter < self.num_parallel_frames:
                    t0 = time.perf_counter()
                    _, frame = self.input_src.read()
                    if frame is not None:
                        frame = self.preprocess_func(frame)
                        img = self.generate_nn_frame(frame)
                        self.frame_queue.put(frame)
                        nn_imgs.append(img)
                        time_used = time.perf_counter() - t0
                        remaining_time = self.frame_lock_time - time_used
                        sleep_time = (
                            remaining_time - 0.0006
                        )  # This is a value set to compensate for the sleep function not being accurate
                        if remaining_time > 0:
                            self.frame_event.wait(sleep_time)
                        counter += 1

                for _ in range(self.num_parallel_frames):
                    self.nn_in_queue.send(nn_imgs.popleft())

                self.grab_frame_flag = False
                self.frame_ready_flag = True
            else:
                time.sleep(0.001)

    def process_frame(self):
        while True:
            if self.frame_ready_flag:
                self.frame_ready_flag = False
                self.grab_frame_flag = True
                for _ in range(self.num_parallel_frames):
                    raw_nn_results = self.nn_out_queue.get()
                    nn_results = self.postprocess_func(raw_nn_results)
                    self.nn_queue.put(nn_results)
                self.nn_result_ready_flag.value = 1
            else:
                time.sleep(0.001)

    def run(self):
        self.grab_frame_thread.start()
        self.process_frame_thread.start()
        self.display_process.start()
        self.grab_frame_thread.join()
        self.process_frame_thread.join()
        self.display_process.join()
