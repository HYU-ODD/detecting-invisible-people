import os
import sys

'''
    after executes this command in darknet,

    $ ./darknet detector test data/obj.data cfg/yolov3.cfg \
        backup/yolov3_14000.weights < ./test.txt > result.txt
    
    convert result.txt to (left, top, right, bottom) format
'''

yolo_pred = '/home/adriv/darknet/night_70.txt'
converted = '/home/adriv/detect-invisible/customAPI/night_70/' # saving path

if __name__ == '__main__':
    if not os.path.exists(converted):
        os.makedirs(converted)

    with open(yolo_pred, 'r') as pred:
        file_path = None
        frame = 0

        while True:
            line = pred.readline()
            if line is None or line == '': break
            splited = line.split()
            
            if len(splited) < 4: continue
            
            if splited[0] == 'Enter':
                file_path = splited[3].split(':')[0]
                frame = file_path.split('.')[0].split('/')[-1]
                with open(converted + frame + '.txt', 'w') as f: pass
            
            elif splited[0] == 'Bounding':
                left = splited[2].split('=')[1].split(',')[0]
                top = splited[3].split('=')[1].split(',')[0]
                right = splited[4].split('=')[1].split(',')[0]
                bot = splited[5].split('=')[1].split(',')[0]
                
                with open(converted + frame + '.txt', 'a') as f:
                    f.write(left + " " + top + " " + right + " " + bot + "\n")
            

    