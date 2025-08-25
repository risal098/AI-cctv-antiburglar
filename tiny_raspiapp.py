import cv2
import torch
from yolox.exp import get_exp
from yolox.utils import fuse_model, postprocess
from yolox.data.data_augment import preproc
from yolox.utils.visualize import vis
from yolox.models import YOLOX
import numpy as np
import time
import os
from datetime import datetime

if not os.path.exists("human_outputs"):
    os.makedirs("human_outputs")


def get_epoch_current():
    epoch_time = time.time()
    return int(epoch_time)


def load_model(exp_file, ckpt_path, device="cpu"):
    exp = get_exp(exp_file, None)

    # ✅ Ensure class names are defined
    if not hasattr(exp, "class_names"):
        exp.class_names = [
            "person"
        ]

    model = exp.get_model().to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = fuse_model(model)
    return model, exp


def run_inference(model, exp, image, device="cpu", conf=0.3, nms=0.45):
    img, ratio = preproc(image, exp.test_size)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)

    with torch.no_grad():
        outputs = model(img)
        outputs = postprocess(outputs, exp.num_classes, conf, nms)

    return outputs, ratio


def draw_result(image, outputs, ratio, class_names,conf):
    if outputs[0] is None:
        return image

    output = outputs[0]

    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()

    bboxes = output[:, 0:4]
    scores = output[:, 4] * output[:, 5]
    cls_ids = output[:, 6]

    bboxes /= ratio

    for i, cls_id in enumerate(cls_ids):
        if int(cls_id) != 0:  # Only detect "person" (class_id = 0)
            continue
        label = class_names[int(cls_id)] if class_names else str(int(cls_id))
        print(f"[INFO] Detected class ID: {int(cls_id)} -> '{label}'")

    return vis(image, bboxes, scores, cls_ids, conf=conf, class_names=class_names)


def save_human_image(image, output_path="human_outputs/detected_{}.jpg"):
    timestamp = int(time.time())
    output_file = output_path.format(timestamp)
    cv2.imwrite(output_file, image)
    print(f"Saved image of detected human to {output_file}")


def main_service():
    # cam_mode = video_path
    # config
    model_path = "tiny_gabung.pth"#"tiny_gabung.pth"   mergegabung110nano.pth
    exp_path ="exps/example/custom/tiny_gabung.py" #"exps/example/custom/tiny_gabung.py"  exps/example/custom/bisconfig.py
    device = "cpu"
    MIN_AREA = 100
    MAX_AREA = 100000
    CONF_THRESHOLD = 0.5
    outputs = [None]
    last_yolo_time = 0
    flaghuman = 0
    interval_capture = 7  # in seconds
    cam_flag = 0
    capdict = {
        "0": {
            "source": "rtsp://admin:123456@192.168.1.253:554/chID=1&streamType=main&linkType=tcp", #changeable
            "flaghuman": 0,
            "last_yolo_time": 0,
            "prev_gray": None,
            "outputs": [None]
        },
        "1": {
            "source": "rtsp://admin:123456@192.168.1.253:554/chID=2&streamType=main&linkType=tcp", #changeable
            "flaghuman": 0,
            "last_yolo_time": 0,
            "prev_gray": None,
            "outputs": [None]
        }
    }
    # "rtsp://admin:Admin123@192.168.1.251:554/h264Preview_01_main"
    # "rtsp://admin:123456@192.168.1.253:554/chID=1&streamType=main&linkType=tcp"
    # [prev_gray,flaghuman,last_yolo_time,outputs]

    camlist=[]
    for cam in capdict:
        # print(capdict[cam])
        cap = cv2.VideoCapture(capdict[cam]["source"])
        ret, prev_frame = cap.read()
        prev_frame=cv2.flip(prev_frame, -1)
        prev_frame=cv2.resize(prev_frame,(416,416))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        capdict[cam]["prev_gray"] = prev_gray
        camlist.append(cap)

    print("tebak")
    model, exp = load_model( exp_path, model_path,device)
    print("agama")
    while True:

        #cap = cv2.VideoCapture(capdict[str(cam_flag)]["source"])
        ret, frame = camlist[cam_flag].read()

        #cv2.imshow("tebak",frame)
        if not ret:
        		continue
        frame=cv2.flip(frame, -1)
        frame=cv2.resize(frame,(416,416))
        
            
        #print("kamera", str(cam_flag))
        capdict[str(cam_flag)]["prev_gray"], \
            capdict[str(cam_flag)]["flaghuman"], \
            capdict[str(cam_flag)]["last_yolo_time"], \
            capdict[str(cam_flag)]["outputs"] = default_detection(model, exp, device, MIN_AREA, MAX_AREA, CONF_THRESHOLD, frame, interval_capture, ret,
                                                                  capdict[str(cam_flag)]["outputs"],
                                                                  capdict[str(cam_flag)
                                                                          ]["last_yolo_time"],
                                                                  capdict[str(cam_flag)
                                                                          ]["flaghuman"],
                                                                  capdict[str(cam_flag)]["prev_gray"],cam_flag)
        cam_flag += 1
        
        if cam_flag >= len(capdict):
            cam_flag = 0
    cap.release()
    print("Video processing completed.")


def default_detection(model, exp, device, MIN_AREA, MAX_AREA, CONF_THRESHOLD, frame, interval_capture, ret,
                      outputs, last_yolo_time, flaghuman, prev_gray,cam_flag):
    if not ret:
        return
    start=time.time()
    #print("process start",start)
    if flaghuman == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        delta = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        dilated = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        human_sized_motion = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if MIN_AREA < area < MAX_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / float(h)
                if 0.3 < aspect_ratio < 3.5:
                    human_sized_motion = True
        prev_gray = gray.copy()
        if human_sized_motion:
            outputs, ratio = run_inference(model, exp, frame, device, CONF_THRESHOLD)
            print("maybe human",cam_flag)

        # Check if person is detected and save image rndnutech
        if outputs[0] is not None:
            result = draw_result(frame, outputs, ratio, exp.class_names,CONF_THRESHOLD)
            result=cv2.resize(frame,(640,640))
            save_human_image(result)
            flaghuman = 1
            last_yolo_time = get_epoch_current()+interval_capture
    else:
        if get_epoch_current() <= last_yolo_time:
            print("interval activated",cam_flag)
            frame=cv2.resize(frame,(640,640))
            save_human_image(frame,"multicctv/intdetect_"+str(get_epoch_current())+".jpg")
        else:
            outputs, ratio = run_inference(model, exp, frame, device, CONF_THRESHOLD)
            if outputs[0] is not None:
                last_yolo_time = get_epoch_current()+interval_capture
                print("human stil detected"+str(cam_flag)+" =======================")
            else:
                flaghuman = 0
                print("human gone "+str(cam_flag)+" ****************************")
    #print("process done",time.time()-start)
    return prev_gray, flaghuman, last_yolo_time, [None]


main_service()
