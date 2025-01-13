#Put this in the yolov7 folder
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, check_requirements, check_imshow, non_max_suppression,
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer,
    set_logging, increment_path
)
from utils.plots import plot_one_box
from utils.torch_utils import (
    select_device, load_classifier, time_synchronized, TracedModel
)

import json
from collections import defaultdict  # Added to support counting

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = (
        opt.source, opt.weights, opt.view_img, opt.save_txt,
        opt.img_size, not opt.no_trace
    )
    save_img = not opt.nosave and not source.endswith('.txt')  # Save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://')
    )

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # Increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # Half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # Load FP32 model
    stride = int(model.stride.max())  # Model stride
    imgsz = check_img_size(imgsz, s=stride)  # Check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # To FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # Initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # Set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Initialize equipment detection counter
    detected_equipments = defaultdict(int)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # Run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (
            old_img_b != img.shape[0] or
            old_img_h != img.shape[2] or
            old_img_w != img.shape[3]
        ):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres,
            classes=opt.classes, agnostic=opt.agnostic_nms
        )
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # Detections per image
            if webcam:  # Batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # To Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = (
                str(save_dir / 'labels' / p.stem) +
                ('' if dataset.mode == 'image' else f'_{frame}')
            )  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # Detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Add to string

                # Write results and collect detected equipment
                for *xyxy, conf, cls in reversed(det):
                    equipment = names[int(cls)]
                    detected_equipments[equipment] += 1  # Update detection count

                    if save_txt:  # Write to file
                        xywh = (
                            xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn
                        ).view(-1).tolist()  # Normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # Label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{equipment} {conf:.2f}'  # Use equipment name instead of class number
                        plot_one_box(
                            xyxy, im0,
                            label=label,
                            color=colors[int(cls)],
                            line_thickness=1
                        )

            # Print time (inference + NMS)
            print(
                f'{s}Done. '
                f'({1E3 * (t2 - t1):.1f}ms) Inference, '
                f'({1E3 * (t3 - t2):.1f}ms) NMS'
            )

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # New video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # Release previous video writer
                        if vid_cap:  # Video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # Stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path,
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            fps,
                            (w, h)
                        )
                    vid_writer.write(im0)

    # After processing all frames, print and save detected equipment
    print("\n=== Detected Equipment Statistics ===")
    if detected_equipments:
        for equip, count in detected_equipments.items():
            print(f"{equip}: {count} times")
    else:
        print("No equipment detected.")

    # Save detected equipment information to JSON
    video_name = Path(opt.source).stem  # Get video filename without extension
    json_path = save_dir / f"{video_name}_equipment.json"

    # Convert defaultdict to a regular dict
    equipment_dict = dict(detected_equipments)

    # Create JSON data structure
    json_data = {
        "video_path": opt.source,
        "detected_equipments": equipment_dict
    }

    # Write to JSON file
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4, ensure_ascii=False)

    print(f"Detected equipment information has been saved to {json_path}")

    if save_txt or save_img:
        s = (
            f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to "
            f"{save_dir / 'labels'}" if save_txt else ''
        )
        # Uncomment the next line if you want to print where the results are saved
        # print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

    # Return the detected equipment dictionary
    return equipment_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # Update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                equipment = detect()
                strip_optimizer(opt.weights)
        else:
            equipment = detect()

    # Print the final detected equipment for easy access
    print("\n=== Final Detected Equipment ===")
    print(equipment)
