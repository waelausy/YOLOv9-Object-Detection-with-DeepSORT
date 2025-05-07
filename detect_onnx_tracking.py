import argparse
import os
import platform
import sys
from pathlib import Path
import math
import torch
import numpy as np
import onnxruntime as ort
import time
import cv2
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, colorstr, cv2,
                          increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device

def initialize_deepsort():
    # Create the Deep SORT configuration object and load settings from the YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # Initialize the DeepSort tracker
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST,
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=False  # Disable CUDA for DeepSORT for better CPU performance
        )

    return deepsort

deepsort = initialize_deepsort()
data_deque = {}

def classNames():
    cocoClassNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                      "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                      "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
                      "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
                      "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
                      "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                      "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
                      "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
                      "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                      "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
                      "toothbrush"]
    return cocoClassNames
    
names = classNames()

def get_trail(id):
    trail_deque = data_deque[id]
    for i in range(len(trail_deque)):
        if i % 2 == 0:
            drawPoint(trail_deque[i], (255, 0, 0), 4)
    
def drawPoint(point_tuple, color, thickness):
    if point_tuple:
        cv2.circle(im0, point_tuple, thickness, color, -1)

# Setup ONNX Runtime Session
def init_onnx_runtime(onnx_path, providers=['CPUExecutionProvider']):
    """Initialize ONNX Runtime session with the specified model and providers."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = os.cpu_count()  # Use all CPU cores
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
    return session

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # wh padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # Assurez-vous que la valeur est un scalaire ou un tuple de la bonne taille
    if len(im.shape) == 3:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    else:
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color[0])
    return im, r, (dw, dh)

@torch.no_grad()
def run(
        source='0',
        onnx_model='yolov9-e.onnx',
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        imgsz=(416, 416),
        conf_thres=0.5,
        iou_thres=0.45,
        classes=None,  # Filter by class: --classes 0, or --classes 0 2 3
        max_det=1000,
        device='cpu',
        view_img=False,
        nosave=False,
        draw_trails=False,
        line_thickness=2,
        hide_labels=False,
        hide_conf=False,
        half=False,
        vid_stride=1,
        agnostic_nms=False
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_img else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize ONNX Runtime
    session = init_onnx_runtime(onnx_model)
    
    # Get session details
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Get model stride and names
    meta = session.get_modelmeta().custom_metadata_map
    stride = int(meta.get('stride', 32))
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    for path, im, im0s, vid_cap, s in dataset:
        # Utiliser directement les prétraitements de dataset
        im = im.astype(np.float32) / 255.0  # 0 - 255 to 0.0 - 1.0
        
        # Create batch dimension
        if len(im.shape) == 3:
            im = np.expand_dims(im, 0)
        
        # Inference
        start = time.time()
        outputs = session.run(output_names, {input_name: im})
        end = time.time()
        
        # Process predictions
        pred = outputs[0]  # Assuming the first output contains detections
        
        # NMS
        pred = torch.from_numpy(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            im0 = im0s.copy() if webcam else im0s[i].copy()
            
            # Add FPS information
            fps = 1.0 / (end - start)
            s += f' - Inference time: {(end - start) * 1000:.1f}ms - FPS: {fps:.1f}'
            
            # Print string
            frame_details = path if webcam else (f"video {i + 1}/{len(dataset)}")
            LOGGER.info(f"{frame_details} {s}")
            
            # Annotator setup
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f" {n} {names[int(c)]}{'s' * (n > 1)},"  # add to string
                
                # Deep SORT tracking
                xywhs = xyxy2xywh(det[:, 0:4])
                confss = det[:, 4]
                oids = det[:, 5]
                
                # Convert to CPU explicitly if needed
                xywhs = torch.tensor(xywhs.cpu())
                confss = torch.tensor(confss.cpu())
                oids = torch.tensor(oids.cpu())
                
                # Update tracker
                outputs = deepsort.update(xywhs, confss, oids, im0)
                
                # Draw boxes and trails
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confss)):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if draw_trails:
                            # Create new buffer for new object
                            if id not in data_deque:  
                                data_deque[id] = deque(maxlen=64)
                            
                            # Add object coordinates to buffer
                            center = (int((bboxes[0] + bboxes[2]) / 2), int((bboxes[1] + bboxes[3]) / 2))
                            data_deque[id].appendleft(center)
                            get_trail(id)

                        c = int(cls)  # integer class
                        label = None if hide_labels else (f'{id} {names[c]} {conf:.2f}' if hide_conf else f'{id} {names[c]}')
                        annotator.box_label(bboxes, label, color=colors(c, True))
                        
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and path not in dataset.cap:
                    pass  # don't display when using Linux terminal (it causes errors)
                else:
                    cv2.imshow(str(path), im0)
                    cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(str(save_dir / Path(path).name), im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_dir / Path(path).name:
                        vid_path[i] = save_dir / Path(path).name
                        if not vid_path[i].exists():
                            vid_path[i].mkdir(parents=True, exist_ok=True)
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        
                        vid_writer[i] = cv2.VideoWriter(
                            str(save_dir / f"{Path(path).stem}.mp4"), 
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                        )
                    vid_writer[i].write(im0)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model', type=str, default='yolov9-e.onnx', help='ONNX model path')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--draw-trails', action='store_true', help='draw trails')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

def main(opt):
    run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
