import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
import utils.transforms as tf
import numpy as np
import models
#from models import sync_bn
import dataset as ds
from options.options import parser
import torch.nn.functional as F
from matplotlib import pyplot as plt
VGG_MEAN = [103.939, 116.779, 123.68]
best_mIoU = 0

import albumentations as A

#################
def minmax_scale(input_arr):
    """

    :param input_arr:
    :return:
    """
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)
    if max_val != min_val:
        output_arr = (input_arr - min_val) / (max_val - min_val)
    else:
        output_arr = input_arr
    return output_arr


def process_embedded(instance_seg_image):
    for i in range(instance_seg_image.shape[-1]):
        instance_seg_image[:, :, i] = minmax_scale(instance_seg_image[:, :, i])
    color_pallet = [np.array([255, 0, 0]),  # blue
                    np.array([0, 255, 0]),  # green
                    np.array([0, 0, 255]),  # red
                    np.array([0, 255, 255]),  # orange
                    np.array([255, 255, 0]),  # yellow
                    np.array([255, 0, 255]),  # cyan
                    np.array([0, 0, 0])  # magenta
                    ]

    legit_lanes = []
    for instance_seg_indx in range(1, instance_seg_image.shape[2]):
        if np.isnan(instance_seg_image[0, 0, instance_seg_indx]):
            continue
        else:
            legit_lanes.append(instance_seg_indx)

    embedding_image = np.zeros((instance_seg_image.shape[0], instance_seg_image.shape[1], 3), dtype=np.uint8)
    for color_indx, lane_indx in enumerate(legit_lanes):
        instance_seg_image_tmp = np.uint8(instance_seg_image[:, :, lane_indx, np.newaxis] *
                                          np.reshape(color_pallet[color_indx],(1, 3)))
        embedding_image = cv2.addWeighted(embedding_image, 1.0, instance_seg_image_tmp, 1.0, 0)
    return embedding_image


def save_im_pred_visual(im, y_pred, y_gt, im_path, show_gt = False):
    blended = im.copy()
    # blended = cv2.cvtColor(np.uint8(blended), cv2.COLOR_RGB2BGR)

    y_pred_copy = y_pred.copy()
    if show_gt:             # GT visualization
        y_gt_copy = np.zeros(y_pred.shape, dtype=np.float32)
        for i in range (1,5):
            y_gti = y_gt_copy[:,:,i]
            y_gti[y_gt == i] = np.ones(y_gti.shape, dtype=np.float32)[y_gt == i]
            y_gt_copy[:,:,i] = y_gti
        y_pred_copy = y_gt_copy
    embedding_image = (process_embedded(y_pred_copy)).astype(np.uint8)
    blended = cv2.addWeighted(blended, 1, embedding_image, 1, 0)
    # blended[:, :, 0] = np.uint8(np.clip(np.float32(blended[:, :, 0]) + 50 * y_gt, 0, 255))
    # blended[:, :, 1] = np.uint8(np.clip(np.float32(blended[:, :, 1]) + 50 * y_gt, 0, 255))
    # blended[:, :, 2] = np.uint8(np.clip(np.float32(blended[:, :, 2]) + 50 * y_gt, 0, 255))

    cv2.imwrite(im_path, blended)


############3


def main(params):
    global args, best_mIoU
    args = parser.parse_args(params)

    vis_aug = False         # enabling visualization of augmentations
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(gpu) for gpu in args.gpus)
    args.gpus = len(args.gpus)

    if args.no_partialbn:
        sync_bn.Synchronize.init(args.gpus)

    if args.dataset == 'VOCAug' or args.dataset == 'VOC2012' or args.dataset == 'COCO':
        num_class = 21
        ignore_label = 255
        scale_series = [10, 20, 30, 60]
    elif args.dataset == 'Cityscapes':
        num_class = 19
        ignore_label = 255 
        scale_series = [15, 30, 45, 90]
    elif args.dataset == 'ApolloScape':
        num_class = 37 
        ignore_label = 255 
    elif args.dataset == 'CULane':
        num_class = 5
        ignore_label = 255
    else:
        raise ValueError('Unknown dataset ' + args.dataset)

    if args.img_height == 360 and args.img_width == 640:
        model = models.ERFNetF(num_class, partial_bn=not args.no_partialbn)
    elif args.img_height == 208 and args.img_width == 976:
        model = models.ERFNet(num_class, partial_bn=not args.no_partialbn)
    else:
        print ("There are no models for this image size.")
        return
    input_mean = model.input_mean
    input_std = model.input_std
    policies = model.get_optim_policies()
    model = torch.nn.DataParallel(model, device_ids=range(args.gpus)).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print(("=> loading checkpoint '{}'".format(args.resume)))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_mIoU = checkpoint['best_mIoU']
            torch.nn.Module.load_state_dict(model, checkpoint['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch'])))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))


    cudnn.benchmark = True
    cudnn.fastest = True

    # Data loading code
    albumentations_transform = A.Compose([
        A.Resize(args.img_height,args.img_width),
        A.Normalize()
    ],p=1)

    # visualization of augmentations used for training
    if vis_aug:
        albumentations_transform = A.Compose([
            A.Resize(args.img_height+40, args.img_width+70),
            #A.RandomScale(scale_limit=0.09, interpolation=1, always_apply=False, p=0.5),
            A.OneOf([
                A.RandomCrop(args.img_height, args.img_width),
                A.RandomSizedCrop((args.img_height-70,args.img_height-1), args.img_height, args.img_width, w2h_ratio=1.77, interpolation=1, always_apply=False, p=1.0),
                A.Resize(args.img_height, args.img_width)
            ],p=1.0),
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
            A.Downscale(scale_min=0.5, scale_max=0.75, interpolation=1, always_apply=False, p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)
            ],p=0.5),
            A.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
            A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
            A.OneOf([
                A.GaussianBlur(blur_limit=7, always_apply=False, p=0.5),
                A.MotionBlur(blur_limit=7, always_apply=False, p=0.5),
                A.MedianBlur(blur_limit=7, always_apply=False, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, always_apply=False, p=0.5)
            ],p=0.2),
            A.Equalize(mode='cv', by_channels=True, mask=None, mask_params=(), always_apply=False, p=0.2),
            A.RandomGamma(gamma_limit=(80, 120), eps=1e-07, always_apply=False, p=0.5),
            A.JpegCompression(quality_lower=50, quality_upper=100, always_apply=False, p=0.2),
            A.OneOf([
                A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2.5, always_apply=False, p=0.5),
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), blur_value=7, brightness_coefficient=0.7, rain_type=None, always_apply=False, p=0.5),
                A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.4, alpha_coef=0.08, always_apply=False, p=0.5),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, always_apply=False, p=0.5)
            ],p=0.2),
            A.Normalize()
        ],p=1)

    test_loader = torch.utils.data.DataLoader(
        getattr(ds, args.dataset.replace("CULane", "VOCAugFFromVideo") + 'DataSet')(data_list=args.val_list, transform=albumentations_transform), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

    # define loss function (criterion) optimizer and evaluator
    weights = [1.0 for _ in range(5)]
    weights[0] = 0.4
    class_weights = torch.FloatTensor(weights).cuda()
    criterion = torch.nn.NLLLoss(ignore_index=ignore_label, weight=class_weights).cuda()
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))
    optimizer = torch.optim.SGD(policies, args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    evaluator = EvalSegmentation(num_class, ignore_label)

    ### evaluate ###
    if args.val_list == 'train_gt2':
        validate(test_loader, model, criterion, 0, evaluator, show_gt=True)
    else:
        validate(test_loader, model, criterion, 0, evaluator)
    return


def validate(val_loader, model, criterion, iter, evaluator, logger=None, show_gt = False):

    batch_time = AverageMeter()
    losses = AverageMeter()
    IoU = AverageMeter()
    mIoU = 0

    amean = [0.485, 0.456, 0.406]
    astd = [0.229, 0.224, 0.225]

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, img_name) in enumerate(val_loader):

        input_var = torch.autograd.Variable(input, volatile=True)

        if show_gt:
            pred = np.zeros((target.shape[0], 5,target.shape[1],target.shape[2]), dtype=np.float32)
            pred_exist = np.zeros((target.shape[0], 4), dtype=np.float32)
        else:
            # compute output torch.Size([5, 3, 208, 976])
            output, output_exist = model(input_var)

            # measure accuracy and record loss

            output = F.softmax(output, dim=1)

            pred = output.data.cpu().numpy() # BxCxHxW
            pred_exist = output_exist.data.cpu().numpy() # BxO

        input_copy = input.data.cpu().numpy()
        y_copy = target.data.cpu().numpy()
        for cnt in range(len(img_name)):
            splits = [pos for pos, char in enumerate(img_name[cnt]) if char == '/']
            directory = 'predicts/Foresight/' + img_name[cnt][:splits[-1]]
            if not os.path.exists(directory):
                os.makedirs(directory)

            # im = np.uint8(input_copy[cnt,:,:,:].transpose((1, 2, 0)) + VGG_MEAN)
            input_copy[cnt, 0, :, :] = (input_copy[cnt,0,:, :] * astd[0] + amean[0]) * 255
            input_copy[cnt, 1, :, :] = (input_copy[cnt,1,:, :] * astd[1] + amean[1]) * 255
            input_copy[cnt, 2, :, :] = (input_copy[cnt,2,:, :] * astd[2] + amean[2]) * 255
            im = np.uint8(input_copy[cnt,:,:,:].transpose(1, 2, 0))
            y_gt = np.array(y_copy[cnt,:,:]).astype(int)
            y_pred = (pred[cnt] * 255).transpose((1, 2, 0))
            im_path = 'predicts/Foresight/'+img_name[cnt]#.strip().split("/")[-1]
            save_im_pred_visual(im, y_pred, y_gt, im_path, show_gt)

            # file_exist = open('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '.exist.txt'), 'w')
            # for num in range(4):
            #     prob_map = (pred[cnt][num+1]*255).astype(int)
            #     save_img = cv2.blur(prob_map,(9,9))
            #     cv2.imwrite('predicts/vgg_SCNN_DULR_w9'+img_name[cnt].replace('.jpg', '_'+str(num+1)+'_avg.png'), save_img)
            #     if pred_exist[cnt][num] > 0.5:
            #         file_exist.write('1 ')
            #     else:
            #         file_exist.write('0 ')
            # file_exist.close()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t' 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time)))

    print('finished, #test:{}'.format(i) )

    return mIoU


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, n=1):
        if self.val is None:
            self.val = val
            self.sum = val * n
            self.count = n
            self.avg = self.sum / self.count
        else:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count


class EvalSegmentation(object):
    def __init__(self, num_class, ignore_label=None):
        self.num_class = num_class
        self.ignore_label = ignore_label

    def __call__(self, pred, gt):
        assert (pred.shape == gt.shape)
        gt = gt.flatten().astype(int)
        pred = pred.flatten().astype(int)
        locs = (gt != self.ignore_label)
        sumim = gt + pred * self.num_class
        hs = np.bincount(sumim[locs], minlength=self.num_class**2).reshape(self.num_class, self.num_class)
        return hs


def adjust_learning_rate(optimizer, epoch, lr_steps):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # decay = 0.1**(sum(epoch >= np.array(lr_steps)))
    decay = ((1 - float(epoch) / args.epochs)**(0.9))
    lr = args.lr * decay
    decay = args.weight_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']


if __name__ == '__main__':
    params = ['CULane',  # dataset ['VOCAug', 'VOC2012', 'COCO', 'Cityscapes', 'ApolloScape', 'CULane']
              'ERFNetF',  # method ['FCN', 'DeepLab', 'DeepLab3', 'PSPNet', 'ERFNet']
              'train',  # train_list
              'test_img',  # val_list
              '--lr', '0.01',
              '--gpus', '0',
              '--resume', 'trained/_erfnetf_model_12smooth+zones+nozero.pth.tar',
              '--img_height', '360',
              '--img_width', '640',
              '--workers', '0',
              '--batch-size', '3'
              ]

    main(params)
