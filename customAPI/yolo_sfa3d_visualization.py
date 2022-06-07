import os
import sys
import cv2
import argparse

root_dir = os.path.dirname(os.path.realpath(__file__))

def draw_bbox(img, corners, color=(0,0,255)):
    left, top, right, bottom = list(map(int,corners))
    img = cv2.line(img, (left, top), (right, top), color, 2)
    img = cv2.line(img, (right, top), (right, bottom), color, 2)
    img = cv2.line(img, (right, bottom), (left, bottom), color, 2)
    img = cv2.line(img, (left, bottom), (left, top), color, 2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Yolo and SFA3D detections')
    parser.add_argument('--vis', action='store_true', help='show image')
    parser.add_argument('--saved_fn', default='vis', help='directory name for saving')
    parser.add_argument('--save', action='store_true', help='save output image')
    parser.add_argument('--image_dir', default='/home/adriv/Carla/CARLA_0.9.13/PythonAPI/DataGenerator/data/town02_test2/training/image_2/')
    parser.add_argument('--yolo_pred_dir', default='./converted')
    parser.add_argument('--sfa3d_pred_dir', default='/home/adriv/SFA3D/results/fpn_resnet_18/predictions/', help='2d bounding box predictions directory')
    parser.add_argument('--only_yolo', action='store_true', help='visualize only yolo detections')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=360)
    args = parser.parse_args()

    yolo_pred = args.yolo_pred_dir
    save_dir = os.path.join(root_dir, args.saved_fn)

    assert os.path.exists(yolo_pred), "Can't find yolo prediction files, " + yolo_pred
    assert os.path.exists(args.sfa3d_pred_dir), "Can't find SFA3D prediction files, " + args.sfa3d_pred_dir
    assert os.path.exists(args.image_dir), "Can't find image directory, " + args.image_dir

    if args.save and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    pred = os.listdir(yolo_pred)
    pred.sort()

    for f in pred:
        frame = f.split('.')[0]
        
        yolo_pred_path = os.path.join(yolo_pred, f)
        sfa3d_pred_path = os.path.join(args.sfa3d_pred_dir, f)
        img_path = args.image_dir + frame + '.png'  

        assert os.path.exists(img_path), "Can't find " + img_path
        
        yolo_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        sfa3d_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        assert yolo_img is not None and sfa3d_img is not None, "Can't open " + img_path
        
        with open(yolo_pred_path, 'r') as rgb:
            while True:
                line = rgb.readline()
                if line is None or line == '': break
                splited = line.split()
                draw_bbox(yolo_img, splited)
            
        if not args.only_yolo and os.path.exists(sfa3d_pred_path):
            with open(sfa3d_pred_path, 'r') as lidar:
                while True:
                    line = lidar.readline()
                    if line is None or line == '': break
                    splited = line.split()
                    draw_bbox(sfa3d_img, splited[1:])

        yolo_img = cv2.resize(yolo_img, (args.width, args.height))
        yolo_img = cv2.putText(yolo_img, "YOLO", (args.width//2-50, args.height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        res = yolo_img

        if not args.only_yolo:
            sfa3d_img = cv2.resize(sfa3d_img, (args.width, args.height))
            sfa3d_img = cv2.putText(sfa3d_img, "LiDAR", (args.width//2-50, args.height-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            res = cv2.vconcat([yolo_img, sfa3d_img])

        if args.save:
            cv2.imwrite(os.path.join(save_dir, frame+'.jpg'), res)

        if args.vis:
            cv2.imshow('image', res)
            cv2.waitKey(50)

    cv2.destroyAllWindows()


        