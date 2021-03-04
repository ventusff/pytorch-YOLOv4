import os
import cv2
import sys
from tqdm import tqdm
import numpy as np
import torch
import random
import imageio
from models import Yolov4
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect


if __name__ == "__main__":

    output_video = True
    video_dir = 'video'
    if output_video:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

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

        use_device = 1  # -1 for cpu, 0,1,2,... for gpu 0,1,2,...
        if use_device >= 0:
            # model.cuda()
            model.to("cuda:{}".format(use_device))

    if not output_video:
        window_name = "gt vs. pred" if eval_model else "gt"
        cv2.namedWindow(window_name, 0)
        cv2.resizeWindow(window_name, 1000, 500)

    class_names = load_class_names(namesfile)
    # explore on dataset
    # fsplit = os.path.join(dataset_dir, "train.txt")
    fsplit = os.path.join(dataset_dir, "val.txt")
    with open(fsplit, 'r') as f:
        lines = f.read().splitlines()

        if output_video:
            it = 0
            # to_plots = []
            # to_plots_seperate = [[] for _ in range(len(class_names))]
            for name in class_names:
                os.makedirs(os.path.join(video_dir, name))
            pbar = tqdm(range(len(lines)), desc="generating video..")
        else:
            it = random.randint(0, len(lines)-1)

        def condition():
            # nonlocal it
            global it
            if output_video:
                it += 1
                return it < len(lines)
                # return it < 100
            else:
                it = (it+1) % len(lines)
                return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1

        while condition():
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

            to_plot = plot_boxes_cv2(img, gt_boxes, class_names=class_names)

            if eval_model:
                with torch.no_grad():
                    # for i in range(2):  # This 'for' loop is for speed check
                    #                     # Because the first iteration is usually longer
                    #     boxes = do_detect(model, sized, 0.4, 0.6, use_cuda)
                    boxes = do_detect(model, sized, 0.4, 0.6, use_device)
                    pred = plot_boxes_cv2(img, boxes[0], class_names=class_names)
                    to_plot = np.concatenate([to_plot, pred], axis=1)
            
            # to_plot = (255 * np.clip(to_plot, 0, 1)).astype(np.uint8)

            if output_video:
                # to_plots.append(to_plot)
                cv2.imwrite(os.path.join(video_dir, str(it) + ".png"), to_plot)
                for i, name in enumerate(class_names):
                    contains = False
                    for box in gt_boxes:
                        if box[6] == i:
                            contains = True
                            break
                    if contains:
                        # to_plots_seperate[i].append(to_plot)
                        cv2.imwrite(os.path.join(video_dir, name, str(it) + ".png"), to_plot)


                pbar.update(1)
            else:
                cv2.imshow(window_name, to_plot)
                keyCode = cv2.waitKey(0)
                if (keyCode & 0xFF) == ord("q"):
                    cv2.destroyAllWindows()
                    break
                elif (keyCode & 0xFF) == ord("n"):
                    it = (it+100) % len(lines)

        pbar.close()
        ############ video
        # imageio.mimwrite(os.path.join(video_dir, "total.mp4"), to_plots, fps=5, quality=8)
        # for i, name in tqdm(enumerate(class_names), desc="output subvideos..."):
        #     if len(to_plots_seperate[i]) > 0:
        #         imageio.mimwrite(os.path.join(video_dir, "{}.mp4".format(name)), to_plots_seperate[i], fps=5, quality=8)