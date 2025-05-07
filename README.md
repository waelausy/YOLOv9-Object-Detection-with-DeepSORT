<H1 align="center">
YOLOv9 Object Detection with DeepSORT Tracking(ID + Trails) </H1>

### New Features
- Added Label for Every Track
- Code can run on Both (CPU & GPU)
- Video/WebCam/External Camera/IP Stream Supported

## Ready to Use Google Colab
[`Google Colab File`](https://colab.research.google.com/drive/1IivrmAtnhpQ1PSmWsp-G6EnqsUOol9v8?usp=sharing)

## Steps to run Code

- Clone the repository
```
git clone https://github.com/MuhammadMoinFaisal/YOLOv9-DeepSORT-Object-Tracking.git
```
- Goto the cloned folder.
```
cd YOLOv9-DeepSORT-Object-Tracking
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download the pre-trained YOLOv9 model weights
[yolov9-c.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)
[yolov9-e.pt](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt) (Larger model, potentially more accurate but slower)

### Exporting to ONNX (Optional)

If you want to use ONNX models for potentially faster CPU inference or cross-platform deployment, you can convert the `.pt` models to `.onnx` format using the `export.py` script:

```bash
# Example for yolov9-c
python export.py --weights yolov9-c.pt --include onnx --imgsz 640 --opset 12

# Example for yolov9-e
python export.py --weights yolov9-e.pt --include onnx --imgsz 640 --opset 12
```

This will create `yolov9-c.onnx` and `yolov9-e.onnx` files respectively.


- Download sample videos from the Google Drive
```
gdown "https://drive.google.com/uc?id=115RBSjNQ_1zjvKFRsQK2zE8v8BIRrpdy&confirm=t"
gdown "https://drive.google.com/uc?id=1rjBn8Fl1E_9d0EMVtL24S9aNQOJAveR5&confirm=t"
```
```
# for detection only
python detect_dual.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --project runs/detect --name exp_c
# Or with yolov9-e.pt:
python detect_dual.py --weights 'yolov9-e.pt' --source 'your video.mp4' --device 0 --img 640 --project runs/detect --name exp_e

#for detection and tracking
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --project runs/track --name exp_c_track
# Or with yolov9-e.pt:
python detect_dual_tracking.py --weights 'yolov9-e.pt' --source 'your video.mp4' --device 0 --img 640 --project runs/track --name exp_e_track

#for WebCam (using yolov9-c.pt by default)
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 0 --device 0 --project runs/track --name webcam_c

#for External Camera
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 1 --device 0 --project runs/track --name ext_cam_c

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source "your IP Camera Stream URL" --device 0 --project runs/track --name live_c

#for specific class (person, class 0)
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0 --project runs/track --name person_c_track

#for detection and tracking with trails 
python detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --draw-trails --project runs/track --name trails_c

### Note importante pour PyTorch 2.6+

Depuis PyTorch 2.6, le paramètre par défaut `weights_only` de `torch.load()` a été changé à `True` pour des raisons de sécurité. Ce projet utilise `weights_only=False` dans `models/experimental.py` pour permettre le chargement correct des modèles YOLOv9. Cette modification est nécessaire pour la compatibilité avec les versions récentes de PyTorch.

### Running with ONNX Models (CPU Recommended)

If you have exported the models to ONNX format, you can use them with `detect_onnx_tracking.py` or `detect_onnx_simple.py`:

```bash
# For ONNX detection and tracking (e.g., with yolov9-c.onnx)
python detect_onnx_tracking.py --weights yolov9-c.onnx --source 'your_video.mp4' --project runs/track_onnx --name exp_onnx_c --classes 0

# For ONNX detection and tracking (e.g., with yolov9-e.onnx)
python detect_onnx_tracking.py --weights yolov9-e.onnx --source 'your_video.mp4' --img-size 640 --project runs/track_onnx --name exp_onnx_e --classes 0

# For ONNX simple detection (without tracking)
python detect_onnx_simple.py --weights yolov9-c.onnx --source 'your_video.mp4' --project runs/detect_onnx --name exp_simple_onnx
```

- Output file will be created in the ```<project_name>/<experiment_name>``` (e.g. runs/track/exp_c_track) with original filename

### Watch the Complete Step by Step Explanation

- Video Tutorial Link  [`YouTube Link`](https://www.youtube.com/watch?v=Jx6oLBfDxRo)


[![Watch the Complete Tutorial for the Step by Step Explanation](https://img.youtube.com/vi/Jx6oLBfDxRo/0.jpg)]([https://www.youtube.com/watch?v=Jx6oLBfDxRo](https://www.youtube.com/watch?v=Jx6oLBfDxRo))
