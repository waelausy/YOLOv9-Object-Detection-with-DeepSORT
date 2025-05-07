import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_img_size, non_max_suppression, print_args, scale_boxes)
from utils.plots import Annotator, colors

def init_onnx_runtime(onnx_path, providers=['CPUExecutionProvider']):
    """Initialize ONNX Runtime session with the specified model and providers."""
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = os.cpu_count()  # Use all CPU cores
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    session = ort.InferenceSession(onnx_path, sess_options=session_options, providers=providers)
    return session

def get_model_info(session):
    """Get model information from ONNX model metadata."""
    meta = session.get_modelmeta().custom_metadata_map
    stride = int(meta.get('stride', 32))  # Model stride
    names = eval(meta.get('names', '[]'))  # Class names
    if not names:
        names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
    return stride, names

@torch.no_grad()
def run(
        source='0',
        onnx_model='yolov9-e.onnx',
        imgsz=(416, 416),
        conf_thres=0.5,
        iou_thres=0.45,
        max_det=1000,
        classes=None,  # Filter by class: --classes 0, or --classes 0 2 3
        vid_stride=1,
        agnostic_nms=False,
        project=ROOT / 'runs/detect',
        name='exp',
        exist_ok=False,
        line_thickness=2,
        hide_labels=False,
        hide_conf=False,
):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    
    # Directories
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    
    # Initialize ONNX Runtime
    session = init_onnx_runtime(onnx_model)
    
    # Get model info
    stride, names = get_model_info(session)
    
    # Get input and output details
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    
    # Check imgsz
    imgsz = check_img_size(imgsz, s=stride)
    
    # Dataloader
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=True, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=True, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    total_time = 0
    frame_count = 0
    
    for path, im, im0s, vid_cap, s in dataset:
        # Normalize image
        im = im.astype(np.float32) / 255.0
        
        # Create batch dimension if needed
        if len(im.shape) == 3:
            im = np.expand_dims(im, 0)
        
        # Time inference
        start = time.time()
        outputs = session.run(output_names, {input_name: im})
        end = time.time()
        inference_time = end - start
        total_time += inference_time
        frame_count += 1
        
        # Process predictions
        pred = outputs[0]  # First output
        
        # Apply NMS
        pred = torch.from_numpy(pred)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        
        # Process detections
        for i, det in enumerate(pred):  # per image
            if webcam:
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)
            save_path = str(save_dir / p.name)
            s += '%gx%g ' % im.shape[2:]  # print string
            
            # Add fps information
            fps = 1.0 / inference_time
            s += f' - {inference_time*1000:.1f}ms ({fps:.1f} FPS)'
            
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f" {n} {names[int(c)]}{'s' * (n > 1)},"  # add to string
                
                # Draw boxes
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = None if hide_labels else (f'{names[c]} {conf:.2f}' if hide_conf else f'{names[c]}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            # Print time (inference-only)
            LOGGER.info(f'{s}')
            
            # Save results (image with detections)
            im0 = annotator.result()
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            else:  # 'video' or 'stream'
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)
    
    # Print final statistics
    if frame_count > 0:
        avg_time = total_time / frame_count
        avg_fps = frame_count / total_time
        LOGGER.info(f'Average inference time: {avg_time*1000:.1f}ms, Average FPS: {avg_fps:.1f}')
    
    return total_time, frame_count

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model', type=str, default='yolov9-e.onnx', help='ONNX model path')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
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
