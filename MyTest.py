import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from scipy import misc
from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset, test_image

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshots/PraNet_Res2Net/PraNet-19.pth')
parser.add_argument('--img_path', type=str, default=None)

opt = parser.parse_args()

if opt.img_path:
    save_path = os.path.join(os.path.dirname(opt.img_path), 'results/')
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    test_loader = test_image(opt.img_path, opt.testsize)
    image, ori_image, name = test_loader.load_data()
    image = image.cuda()

    res5, res4, res3, res2 = model(image)
    res = res2
    res = F.upsample(res, size=(ori_image.size[1], ori_image.size[0]), mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    misc.imsave(save_path+name, res)

    os.sys.exit(0)

for _data_name in ['CAMO', 'CHAMELEON', 'COD10K']:
    data_path = 'data/TestDataset/{}/'.format(_data_name)
    save_path = './results/PraNet_v3/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = PraNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res5, res4, res3, res2 = model(image)
        res = res2
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        misc.imsave(save_path+name, res)
