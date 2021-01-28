import yaml
import torch
import argparse
import timeit
import numpy as np
import matplotlib.pyplot as plt
#import cv2

from torch.utils import data


from ptsemseg.models import get_model
from ptsemseg.loader import get_loader
from ptsemseg.metrics import runningScore
from ptsemseg.utils import convert_state_dict

torch.backends.cudnn.benchmark = True


def validate(cfg, args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(device)
    print(device)
    print(device)
    print(device)
    print(device)
    print(device)
    print(device)
    
    # Setup Dataloader
    data_loader = get_loader(cfg["data"]["dataset"])
    data_path = cfg["data"]["path"]

    loader = data_loader(
        data_path,
        split=cfg["data"]["val_split"],
        is_transform=True,
        img_size=(cfg["data"]["img_rows"], cfg["data"]["img_cols"]),
    )

    n_classes = loader.n_classes

    valloader = data.DataLoader(loader, batch_size=cfg["training"]["batch_size"], num_workers=8)
    running_metrics = runningScore(n_classes)

    # Setup Model

    model = get_model(cfg["model"], n_classes).to(device)
    state = convert_state_dict(torch.load(args.model_path)["model_state"])
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    for i, (images, labels) in enumerate(valloader):
        start_time = timeit.default_timer()

        #limit the number of runs
        if i >= 2:
            break
        
        images = images.to(device)

        if args.eval_flip:
            outputs = model(images)

            # Flip images in numpy (not support in tensor)
            outputs = outputs.data.cpu().numpy()
            flipped_images = np.copy(images.data.cpu().numpy()[:, :, :, ::-1])
            flipped_images = torch.from_numpy(flipped_images).float().to(device)
            outputs_flipped = model(flipped_images)
            outputs_flipped = outputs_flipped.data.cpu().numpy()
            outputs = (outputs + outputs_flipped[:, :, :, ::-1]) / 2.0

            pred = np.argmax(outputs, axis=1)
        else:
            outputs = model(images)
            pred = outputs.data.max(1)[1].cpu().numpy()
        
        gt = labels.numpy()
        print('labels shape: ', gt.shape, 'max of labels: ', np.max(gt), 'min of labels: ', np.min(gt) )
        #print(gt)
        gtArray = np.array(gt[1,:,:])
        gtArray = np.interp(gtArray, (gtArray.min(), gtArray.max()), (0, 18)) 
        print('gtArray shape: ', gtArray.shape, 'max of gtArray: ', np.max(gtArray), 'min of gtArray: ', np.min(gtArray) )
        
        name = 'labels '+str(i)
        plt.imshow(gtArray)
        #plt.colorbar()
        plt.savefig(name, dpi=200)
        plt.show()
        
        #here we want to save prediction? what is its shape and what is shape of image?
        print('image shape: ', images.shape)
        imagesTemp = images.cpu().numpy()
        imageArray = np.array(imagesTemp[1,:,:,:])
        print('image shape: ', imageArray.shape)
        sz = imageArray.shape
        imageArray = np.reshape(imageArray, (sz[1], sz[2], 3))
        imageArray = imageArray[0:85, 0:170, :]
        #imageArray = cv2.resize(imageArray, dsize = (512, 256), interpolation=cv2.INTER_CUBIC)
        name = 'image'+str(i)
        plt.imshow(imageArray)
        #plt.colorbar()
        plt.savefig(name, dpi=200)
        plt.show()
        
        print('prediction shape: ', pred.shape)
        predArray = np.array(pred[1,:,:])
        print('prediction shape: ', predArray.shape)
        sz = predArray.shape
        #predArray = np.reshape(predArray, (sz[1], sz[2], 3))
        name = 'prediction'+str(i)
        plt.imshow(predArray)
        plt.colorbar()
        plt.savefig(name, dpi=200)
        plt.show()
        
        
        

        if args.measure_time:
            elapsed_time = timeit.default_timer() - start_time
            print(
                "Inference time \
                  (iter {0:5d}): {1:3.5f} fps".format(
                    i + 1, pred.shape[0] / elapsed_time
                )
            )
        running_metrics.update(gt, pred)

    score, class_iou = running_metrics.get_scores()

    for k, v in score.items():
        print(k, v)

    for i in range(n_classes):
        print(i, class_iou[i])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparams")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/fcn8s_pascal.yml",
        help="Config file to be used",
    )
    parser.add_argument(
        "--model_path",
        nargs="?",
        type=str,
        default="fcn8s_pascal_1_26.pkl",
        help="Path to the saved model",
    )
    parser.add_argument(
        "--eval_flip",
        dest="eval_flip",
        action="store_true",
        help="Enable evaluation with flipped image |\
                              True by default",
    )
    parser.add_argument(
        "--no-eval_flip",
        dest="eval_flip",
        action="store_false",
        help="Disable evaluation with flipped image |\
                              True by default",
    )
    parser.set_defaults(eval_flip=True)

    parser.add_argument(
        "--measure_time",
        dest="measure_time",
        action="store_true",
        help="Enable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.add_argument(
        "--no-measure_time",
        dest="measure_time",
        action="store_false",
        help="Disable evaluation with time (fps) measurement |\
                              True by default",
    )
    parser.set_defaults(measure_time=True)

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    validate(cfg, args)
