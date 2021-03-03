import os
import cv2
import sys
import numpy as np
import torch
import random
from models import Yolov4
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect


if __name__ == "__main__":

    eval_model = False
    if len(sys.argv) == 6:
        n_classes = int(sys.argv[1])
        namesfile = sys.argv[2]
        dataset_dir = sys.argv[3]
        width = int(sys.argv[4])    # original size of the image
        height = int(sys.argv[5])
    elif len(sys.argv) == 7:
        n_classes = int(sys.argv[1])
        namesfile = sys.argv[2]
        dataset_dir = sys.argv[3]
        width = int(sys.argv[4])
        height = int(sys.argv[5])
        weightfile = sys.argv[6]
        eval_model = True
    else:
        print('Usage: ')
        print('  python models.py [num_classes] [names_file] [datasetdir] [width] [height] ([pretrained])')

    if eval_model:
        model = Yolov4(yolov4conv137weight=None, n_classes=n_classes, inference=True)

        pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
        model.load_state_dict(pretrained_dict)

        use_cuda = True
        if use_cuda:
            model.cuda()

    window_name = "gt vs. pred" if eval_model else "gt"
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, 1000, 500)

    with torch.no_grad():

        # explore on dataset
        fsplit = os.path.join(dataset_dir, "train.txt")
        # fsplit = os.path.join(dataset_dir, "val.txt")
        with open(fsplit, 'r') as f:
            lines = f.read().splitlines()
            it = random.randint(0, len(lines)-1)

            while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
                it = (it+1) % len(lines)

                splits = lines[it].split(' ')
                imgfile = splits[0]
                gt_boxes_ = splits[1:]
                gt_boxes = []
                for box in gt_boxes_:
                    box_splits = box.split(',')
                    box_s = [int(b) for b in box_splits]
                    x1, y1, x2, y2, id = box_s
                    gt_boxes.append(
                        [x1 / float(width), 
                        y1 / float(height),
                        x2 / float(width),
                        y2 / float(height),
                        None, "", id])

                img = cv2.imread(os.path.join(dataset_dir, imgfile))

                # Inference input size is 416*416 does not mean training size is the same
                # Training size could be 608*608 or even other sizes
                # Optional inference sizes:
                #   Hight in {320, 416, 512, 608, ... 320 + 96 * n}
                #   Width in {320, 416, 512, 608, ... 320 + 96 * m}
                sized = cv2.resize(img, (width, height))
                sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

                class_names = load_class_names(namesfile)

                to_plot = plot_boxes_cv2(img, gt_boxes, class_names=class_names)

                if eval_model:
                    # for i in range(2):  # This 'for' loop is for speed check
                    #                     # Because the first iteration is usually longer
                    #     boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

                    boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)

                    pred = plot_boxes_cv2(img, boxes[0], class_names=class_names)

                    to_plot = np.concatenate([to_plot, pred], axis=1)
                
                cv2.imshow(window_name, to_plot)
                keyCode = cv2.waitKey(0)
                if (keyCode & 0xFF) == ord("q"):
                    cv2.destroyAllWindows()
                    break
                elif (keyCode & 0xFF) == ord("n"):
                    it = (it+100) % len(lines)

            