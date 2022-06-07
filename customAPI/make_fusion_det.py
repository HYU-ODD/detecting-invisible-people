import os
import sys
import argparse
import cv2

root_dir = os.path.dirname(os.path.realpath(__file__))

def calc_iou(bbox1, bbox2):
    left1, top1, right1, bot1 = list(map(int,bbox1))
    left2, top2, right2, bot2 = list(map(int,bbox2))

    x_len = max(0, min(right1, right2) - max(left1, left2))
    y_len = max(0, min(bot1, bot2) - max(top1, top2))

    intersection = x_len * y_len
    union = (right1-left1) * (bot1-top1) + (right2-left2) * (bot2-top2) - intersection

    return float(intersection) / float(union)


def draw_bbox(img, corners, color=(0,0,255)):
    left, top, right, bottom = list(map(int,corners))
    img = cv2.line(img, (left, top), (right, top), color, 2)
    img = cv2.line(img, (right, top), (right, bottom), color, 2)
    img = cv2.line(img, (right, bottom), (left, bottom), color, 2)
    img = cv2.line(img, (left, bottom), (left, top), color, 2)


def bbox_to_str(frame, bbox):
    left, top, right, bottom = list(map(int,bbox))
    width = right - left + 1
    height = bottom - top + 1

    return str(int(frame)) + ',-1,' + bbox[0] + ',' + bbox[1] + ',' + str(width) + ',' + str(height) + ',1\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='make det.txt with yolo and lidar detection in MOT format')
    parser.add_argument('--yolo_pred_dir', default='./converted')
    parser.add_argument('--sfa3d_pred_dir', default='/home/adriv/SFA3D/results/fpn_resnet_18/predictions/')
    parser.add_argument('--iou_thresh', type=float, default=0.2, help='IoU threshold')
    parser.add_argument('--save_fn', default='det.txt')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--image_dir', default='/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data/town02_dark2/training/image_2/')
    parser.add_argument('--yolo_only_det', action='store_true', help='make det.txt with only yolo predictions')
    args = parser.parse_args()

    assert os.path.exists(args.yolo_pred_dir), "Can't find yolo prediction files, " + args.yolo_pred_dir
    assert os.path.exists(args.sfa3d_pred_dir), "Can't find SFA3D prediction files, " + args.sfa3d_pred_dir
    assert os.path.exists(args.image_dir), "Can't find image directory, " + args.image_dir

    pred = os.listdir(args.yolo_pred_dir)
    pred.sort()

    with open(os.path.join(root_dir, args.save_fn), 'w') as new_det:
        for f in pred:
            frame = f.split('.')[0]

            yolo_pred_path = os.path.join(args.yolo_pred_dir, f)
            sfa3d_pred_path = os.path.join(args.sfa3d_pred_dir, f)
            img_path = args.image_dir + frame + '.png' 

            assert os.path.exists(img_path), "Can't find " + img_path
            if args.vis:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

            yolo_det = []
            with open(yolo_pred_path, 'r') as rgb:
                while True:
                    line = rgb.readline()
                    if line is None or line == '': break
                    splited = line.split()
                    yolo_det.append(splited)
            
            lidar_det = []
            if os.path.exists(sfa3d_pred_path):
                with open(sfa3d_pred_path , 'r') as lidar:
                    while True:
                        line = lidar.readline()
                        if line is None or line == '': break
                        splited = line.split()
                        lidar_det.append(splited[1:])
            
            for yolo_bbox in yolo_det:
                overlap = False

                for lidar_bbox in lidar_det:
                    iou = calc_iou(yolo_bbox, lidar_bbox)
                    if iou > args.iou_thresh and not args.yolo_only_det: 
                        overlap = True
                        break

                if not overlap: 
                    new_det.write(bbox_to_str(frame, yolo_bbox))
                    if args.vis:
                        draw_bbox(img, yolo_bbox, (255,255,255))
            
            if not args.yolo_only_det:
                for lidar_bbox in lidar_det:
                    new_det.write(bbox_to_str(frame, lidar_bbox))
                    if args.vis:
                        draw_bbox(img, lidar_bbox, (0,255,0))

            if args.vis:
                cv2.imshow('result', img)
                cv2.waitKey(50)

    cv2.destroyAllWindows()