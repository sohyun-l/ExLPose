import os
import argparse
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets

from train_config import cfg
from test_config_all import cfg_test
from test_config_normal import cfg_test_normal
from test_config_hard import cfg_test_hard
from test_config_extreme import cfg_test_extreme
from pycocotools.coco_custom import COCO
from pycocotools.cocoeval_custom import COCOeval
from utils.logger import Logger
from utils.evaluation import accuracy, AverageMeter, final_preds
from utils.misc import save_model, adjust_learning_rate
from utils.osutils import mkdir_p, isfile, isdir, join
from utils.transforms import fliplr, flip_back
from utils.imutils import im_to_numpy, im_to_torch

from networks import network 
from dataloader.loader_training_pair import TrainingData
from dataloader.loader_eval_LL import EvalLLData
from dataloader.loader_eval_WL import EvalWLData
import wandb
from datetime import datetime
import cv2
import json
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

def get_optimizer_params(modules, lr, weight_decay=0.0005, double_bias_lr=True, base_weight_factor=1):
    weights = []
    biases = []
    base_weights = []
    base_biases = []
    if isinstance(modules, list):
        for module in modules:
            for key, value in dict(module.named_parameters()).items():
                if value.requires_grad:
                    if 'fc' in key or 'score' in key:
                        if 'bias' in key:
                            biases += [value]
                        else:
                            weights += [value]
                    else:
                        if 'bias' in key:
                            base_biases += [value]
                        else:
                            base_weights += [value]
    else:
        module = modules
        for key, value in dict(module.named_parameters()).items():
            if value.requires_grad:
                if 'fc' in key or 'score' in key:
                    if 'bias' in key:
                        biases += [value]
                    else:
                        weights += [value]
                else:
                    if 'bias' in key:
                        base_biases += [value]
                    else:
                        base_weights += [value]
    if base_weight_factor:
        params = [
            {'params': weights, 'lr': lr, 'weight_decay': weight_decay},
            {'params': biases, 'lr': lr},
            {'params': base_weights, 'lr': lr * base_weight_factor, 'weight_decay': weight_decay},
            {'params': base_biases, 'lr': lr * base_weight_factor},
        ]
    else:
        params = [
            {'params': base_weights + weights, 'lr': lr, 'weight_decay': weight_decay},
            {'params': base_biases + biases, 'lr': lr },
        ]
    return params

def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def plot_images_to_wandb(images: list, name: str, step=None):
    # images are should be list of RGB images tensors in shape (C, H, W)
    images = vutils.make_grid(images, normalize=True, range=(-2.11785, 2.64005))

    if images.dim() == 3:
        images = images.permute(1, 2, 0)
    images = images.detach().cpu().numpy()
    images = wandb.Image(images, caption=name)

    wandb.log({name: images}, step=step)

def main(args):
    # create checkpoint dir
    if not isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-{now}'

    wandb.init(project="Dark_project", name=f'{run_name}', entity="poseindark")

    # create model
    model = network.__dict__[cfg.model](cfg.output_shape, cfg.num_class, in_features=0, num_conditions=2, pretrained=True)
    model = torch.nn.DataParallel(model).cuda()


    wl_idx = 1

    # define loss function (criterion) and optimizer
    criterion1 = torch.nn.MSELoss().cuda() # for Global loss
    criterion2 = torch.nn.MSELoss(reduce=False).cuda() # for refine loss
    lupi_criterion = torch.nn.MSELoss().cuda()

    params = get_optimizer_params(model, cfg.lr, weight_decay=cfg.weight_decay)
    optimizer = torch.optim.Adam(params) 


    wandb.config = {
        "learning_rate": cfg.lr,
        "weight_decay":cfg.weight_decay,
        "epochs": args.epochs,
        "batch_size": cfg.batch_size
        }

    if args.resume:
        if isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            pretrained_dict = checkpoint['state_dict']
            model.load_state_dict(pretrained_dict)
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            # logger = Logger(join(args.checkpoint, 'log.txt'), resume=True)
            logger = Logger(join(args.checkpoint, 'log.txt'))
            logger.set_names(['Epoch', 'LR', 'Train Loss'])
            args.start_epoch = 0
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:        
        logger = Logger(join(args.checkpoint, 'log.txt'))
        logger.set_names(['Epoch', 'LR', 'Train Loss'])

    cudnn.benchmark = True
    print('    Total params: %.2fMB' % (sum(p.numel() for p in model.parameters())/(1024*1024)*4))

    train_loader = torch.utils.data.DataLoader(
        TrainingData(cfg, cfg),
        batch_size=cfg.batch_size*args.num_gpus, shuffle=True,
        num_workers=args.workers, pin_memory=True) 


    length = len(train_loader)
    
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, cfg.lr_dec_epoch, cfg.lr_gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr)) 
        wandb.log({"learning_rate": lr}, step=epoch)

        # train for one epoch
        train_loss = train(epoch, length, train_loader, model, [criterion1, criterion2, lupi_criterion], optimizer)
        print('train_loss: ',train_loss)
        wandb.log({"train_loss": train_loss}, step=epoch)

        # append logger file
        logger.append([epoch + 1, lr, train_loss])

        if epoch > 2:
            save_model({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, checkpoint=args.checkpoint)

        model.eval()

        test_loader = torch.utils.data.DataLoader(
        EvalLLData(cfg_test_normal, train=False),
        batch_size=128*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

        print('testing...')
        full_result = []
        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
                ll_idx = 0
                # compute output
                f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                score_map = refine_output.data.cpu() 
                score_map = score_map.numpy()

                f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1,2,0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2,0,1)))
                    for (q, w) in cfg_test_normal.symmetry:
                        fscore[q], fscore[w] = fscore[w], fscore[q] 
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

                ids = meta['imgID'].numpy()
                det_scores = meta['det_scores']
                for b in range(inputs.size(0)):
                    details = meta['augmentation_details']
                    single_result_dict = {}
                    single_result = []
                    
                    single_map = score_map[b]
                    r0 = single_map.copy()
                    r0 /= 255
                    r0 += 0.5
                    v_score = np.zeros(14)
                    for p in range(14): 
                        single_map[p] /= np.amax(single_map[p])
                        border = 10
                        dr = np.zeros((cfg_test_normal.output_shape[0] + 2*border, cfg_test_normal.output_shape[1]+2*border))
                        dr[border:-border, border:-border] = single_map[p].copy()
                        dr = cv2.GaussianBlur(dr, (21, 21), 0)
                        lb = dr.argmax()
                        y, x = np.unravel_index(lb, dr.shape)
                        dr[y, x] = 0
                        lb = dr.argmax()
                        py, px = np.unravel_index(lb, dr.shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, cfg_test_normal.output_shape[1] - 1))
                        y = max(0, min(y, cfg_test_normal.output_shape[0] - 1))
                        resy = float((4 * y + 2) / cfg_test_normal.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                        resx = float((4 * x + 2) / cfg_test_normal.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                        single_result.append(resx)
                        single_result.append(resy)
                        single_result.append(1)   
                    if len(single_result) != 0:
                        single_result_dict['image_id'] = int(ids[b])
                        single_result_dict['category_id'] = 1
                        single_result_dict['keypoints'] = single_result
                        single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                        full_result.append(single_result_dict)

        result_path = 'result_LL_normal'
        if not isdir(result_path):
            mkdir_p(result_path)
        result_file = os.path.join(result_path, 'result.json')
        with open(result_file,'w') as wf:
            json.dump(full_result, wf)

        eval_gt = COCO(cfg_test_normal.ori_gt_path)
        eval_dt = eval_gt.loadRes(result_file)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        result = cocoEval.summarize()

        wandb.log({"LL_normalsplit_AP": result[0]})

        test_loader = torch.utils.data.DataLoader(
        EvalLLData(cfg_test_hard, train=False),
        batch_size=128*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

        print('testing...')
        full_result = []
        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
                ll_idx = 0
                # compute output
                f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                score_map = refine_output.data.cpu() 
                score_map = score_map.numpy()

                f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1,2,0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2,0,1)))
                    for (q, w) in cfg_test_hard.symmetry:
                        fscore[q], fscore[w] = fscore[w], fscore[q] 
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

                ids = meta['imgID'].numpy()
                det_scores = meta['det_scores']
                for b in range(inputs.size(0)):
                    details = meta['augmentation_details']
                    single_result_dict = {}
                    single_result = []
                    
                    single_map = score_map[b]
                    r0 = single_map.copy()
                    r0 /= 255
                    r0 += 0.5
                    v_score = np.zeros(14)
                    for p in range(14): 
                        single_map[p] /= np.amax(single_map[p])
                        border = 10
                        dr = np.zeros((cfg_test_hard.output_shape[0] + 2*border, cfg_test_hard.output_shape[1]+2*border))
                        dr[border:-border, border:-border] = single_map[p].copy()
                        dr = cv2.GaussianBlur(dr, (21, 21), 0)
                        lb = dr.argmax()
                        y, x = np.unravel_index(lb, dr.shape)
                        dr[y, x] = 0
                        lb = dr.argmax()
                        py, px = np.unravel_index(lb, dr.shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, cfg_test_hard.output_shape[1] - 1))
                        y = max(0, min(y, cfg_test_hard.output_shape[0] - 1))
                        resy = float((4 * y + 2) / cfg_test_hard.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                        resx = float((4 * x + 2) / cfg_test_hard.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                        single_result.append(resx)
                        single_result.append(resy)
                        single_result.append(1)   
                    if len(single_result) != 0:
                        single_result_dict['image_id'] = int(ids[b])
                        single_result_dict['category_id'] = 1
                        single_result_dict['keypoints'] = single_result
                        single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                        full_result.append(single_result_dict)

        result_path = 'result_LL_hard'
        if not isdir(result_path):
            mkdir_p(result_path)
        result_file = os.path.join(result_path, 'result.json')
        with open(result_file,'w') as wf:
            json.dump(full_result, wf)

        eval_gt = COCO(cfg_test_hard.ori_gt_path)
        eval_dt = eval_gt.loadRes(result_file)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        result = cocoEval.summarize()
        wandb.log({"LL_hardsplit_AP": result[0]})

        test_loader = torch.utils.data.DataLoader(
        EvalLLData(cfg_test_extreme, train=False),
        batch_size=128*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

        print('testing...')
        full_result = []
        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())

                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
                ll_idx = 0
                # compute output
                f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                score_map = refine_output.data.cpu() 
                score_map = score_map.numpy()


                f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1,2,0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2,0,1)))
                    for (q, w) in cfg_test_extreme.symmetry:
                        fscore[q], fscore[w] = fscore[w], fscore[q] 
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

                ids = meta['imgID'].numpy()
                det_scores = meta['det_scores']
                for b in range(inputs.size(0)):
                    details = meta['augmentation_details']
                    single_result_dict = {}
                    single_result = []
                    
                    single_map = score_map[b]
                    r0 = single_map.copy()
                    r0 /= 255
                    r0 += 0.5
                    v_score = np.zeros(14)
                    for p in range(14): 
                        single_map[p] /= np.amax(single_map[p])
                        border = 10
                        dr = np.zeros((cfg_test_extreme.output_shape[0] + 2*border, cfg_test_extreme.output_shape[1]+2*border))
                        dr[border:-border, border:-border] = single_map[p].copy()
                        dr = cv2.GaussianBlur(dr, (21, 21), 0)
                        lb = dr.argmax()
                        y, x = np.unravel_index(lb, dr.shape)
                        dr[y, x] = 0
                        lb = dr.argmax()
                        py, px = np.unravel_index(lb, dr.shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, cfg_test_extreme.output_shape[1] - 1))
                        y = max(0, min(y, cfg_test_extreme.output_shape[0] - 1))
                        resy = float((4 * y + 2) / cfg_test_extreme.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                        resx = float((4 * x + 2) / cfg_test_extreme.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                        single_result.append(resx)
                        single_result.append(resy)
                        single_result.append(1)   
                    if len(single_result) != 0:
                        single_result_dict['image_id'] = int(ids[b])
                        single_result_dict['category_id'] = 1
                        single_result_dict['keypoints'] = single_result
                        single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                        full_result.append(single_result_dict)

        result_path = 'result_LL_extreme'
        if not isdir(result_path):
            mkdir_p(result_path)
        result_file = os.path.join(result_path, 'result.json')
        with open(result_file,'w') as wf:
            json.dump(full_result, wf)
        # evaluate on COCO
        
        eval_gt = COCO(cfg_test_extreme.ori_gt_path)
        eval_dt = eval_gt.loadRes(result_file)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        result = cocoEval.summarize()

        wandb.log({"LL_extremesplit_AP": result[0]})


        test_loader = torch.utils.data.DataLoader(
        EvalLLData(cfg_test, train=False),
        batch_size=128*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

        print('testing...')
        full_result = []
        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())

                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
                ll_idx = 0
                # compute output
                f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                score_map = refine_output.data.cpu() 
                score_map = score_map.numpy()


                f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var, ll_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1,2,0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2,0,1)))
                    for (q, w) in cfg_test.symmetry:
                        fscore[q], fscore[w] = fscore[w], fscore[q] 
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

                ids = meta['imgID'].numpy()
                det_scores = meta['det_scores']
                for b in range(inputs.size(0)):
                    details = meta['augmentation_details']
                    single_result_dict = {}
                    single_result = []
                    
                    single_map = score_map[b]
                    r0 = single_map.copy()
                    r0 /= 255
                    r0 += 0.5
                    v_score = np.zeros(14)
                    for p in range(14): 
                        single_map[p] /= np.amax(single_map[p])
                        border = 10
                        dr = np.zeros((cfg_test.output_shape[0] + 2*border, cfg_test.output_shape[1]+2*border))
                        dr[border:-border, border:-border] = single_map[p].copy()
                        dr = cv2.GaussianBlur(dr, (21, 21), 0)
                        lb = dr.argmax()
                        y, x = np.unravel_index(lb, dr.shape)
                        dr[y, x] = 0
                        lb = dr.argmax()
                        py, px = np.unravel_index(lb, dr.shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, cfg_test.output_shape[1] - 1))
                        y = max(0, min(y, cfg_test.output_shape[0] - 1))
                        resy = float((4 * y + 2) / cfg_test.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                        resx = float((4 * x + 2) / cfg_test.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                        single_result.append(resx)
                        single_result.append(resy)
                        single_result.append(1)   
                    if len(single_result) != 0:
                        single_result_dict['image_id'] = int(ids[b])
                        single_result_dict['category_id'] = 1
                        single_result_dict['keypoints'] = single_result
                        single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                        full_result.append(single_result_dict)

        result_path = 'result_LL_all'
        if not isdir(result_path):
            mkdir_p(result_path)
        result_file = os.path.join(result_path, 'result.json')
        with open(result_file,'w') as wf:
            json.dump(full_result, wf)

        eval_gt = COCO(cfg_test.ori_gt_path)
        eval_dt = eval_gt.loadRes(result_file)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        result = cocoEval.summarize()


        wandb.log({"LL_all_AP": result[0]})

        test_loader = torch.utils.data.DataLoader(
        EvalWLData(cfg_test, train=False),
        batch_size=128*args.num_gpus, shuffle=False,
        num_workers=args.workers, pin_memory=True) 

        print('testing...')
        full_result = []
        for i, (inputs, meta) in tqdm(enumerate(test_loader)):
            with torch.no_grad():
                input_var = torch.autograd.Variable(inputs.cuda())
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    flip_inputs[i] = im_to_torch(finp)
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())
                wl_idx = 1
                # compute output
                f0, f1, f2, f3, f4, global_outputs, refine_output = model(input_var, wl_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
                score_map = refine_output.data.cpu()
                score_map = score_map.numpy()

                f0, f1, f2, f3, f4, flip_global_outputs, flip_output = model(flip_input_var, wl_idx * torch.ones(flip_input_var.shape[0], dtype=torch.long).cuda())
                flip_score_map = flip_output.data.cpu()
                flip_score_map = flip_score_map.numpy()

                for i, fscore in enumerate(flip_score_map):
                    fscore = fscore.transpose((1,2,0))
                    fscore = cv2.flip(fscore, 1)
                    fscore = list(fscore.transpose((2,0,1)))
                    for (q, w) in cfg_test.symmetry:
                        fscore[q], fscore[w] = fscore[w], fscore[q] 
                    fscore = np.array(fscore)
                    score_map[i] += fscore
                    score_map[i] /= 2

                ids = meta['imgID'].numpy()
                det_scores = meta['det_scores']
                for b in range(inputs.size(0)):
                    details = meta['augmentation_details']
                    single_result_dict = {}
                    single_result = []
                    
                    single_map = score_map[b]
                    r0 = single_map.copy()
                    r0 /= 255
                    r0 += 0.5
                    v_score = np.zeros(14)
                    for p in range(14): 
                        single_map[p] /= np.amax(single_map[p])
                        border = 10
                        dr = np.zeros((cfg_test.output_shape[0] + 2*border, cfg_test.output_shape[1]+2*border))
                        dr[border:-border, border:-border] = single_map[p].copy()
                        dr = cv2.GaussianBlur(dr, (21, 21), 0)
                        lb = dr.argmax()
                        y, x = np.unravel_index(lb, dr.shape)
                        dr[y, x] = 0
                        lb = dr.argmax()
                        py, px = np.unravel_index(lb, dr.shape)
                        y -= border
                        x -= border
                        py -= border + y
                        px -= border + x
                        ln = (px ** 2 + py ** 2) ** 0.5
                        delta = 0.25
                        if ln > 1e-3:
                            x += delta * px / ln
                            y += delta * py / ln
                        x = max(0, min(x, cfg_test.output_shape[1] - 1))
                        y = max(0, min(y, cfg_test.output_shape[0] - 1))
                        resy = float((4 * y + 2) / cfg_test.data_shape[0] * (details[b][3] - details[b][1]) + details[b][1])
                        resx = float((4 * x + 2) / cfg_test.data_shape[1] * (details[b][2] - details[b][0]) + details[b][0])
                        v_score[p] = float(r0[p, int(round(y)+1e-10), int(round(x)+1e-10)])                
                        single_result.append(resx)
                        single_result.append(resy)
                        single_result.append(1)   
                    if len(single_result) != 0:
                        single_result_dict['image_id'] = int(ids[b])
                        single_result_dict['category_id'] = 1
                        single_result_dict['keypoints'] = single_result
                        single_result_dict['score'] = float(det_scores[b])*v_score.mean()
                        full_result.append(single_result_dict)

        result_path = 'result_WL'
        if not isdir(result_path):
            mkdir_p(result_path)
        result_file = os.path.join(result_path, 'result.json')
        with open(result_file,'w') as wf:
            json.dump(full_result, wf)
        
        eval_gt = COCO(cfg_test.ori_gt_path)
        eval_dt = eval_gt.loadRes(result_file)
        cocoEval = COCOeval(eval_gt, eval_dt, iouType='keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        result = cocoEval.summarize()

        wandb.log({"WL_AP": result[0]})
        

    logger.close()



def train(epoch, length, train_loader, model, criterions, optimizer):
    # prepare for refine loss


    def ohkm(loss, top_k):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(sub_loss, k=top_k, dim=0, sorted=False)
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / top_k
        ohkm_loss /= loss.size()[0]
        return ohkm_loss
    criterion1, criterion2, lupi_criterion = criterions

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    train_loader_iter = enumerate(train_loader)

    # switch to train mode
    model.train()

    for i in range(length):

        loss_content = 0
        loss_lupi = 0
        ll_idx = 0
        wl_idx = 1

        loss = 0.
        global_loss_record = 0.
        refine_loss_record = 0.
        global_loss_record_wl = 0.
        refine_loss_record_wl = 0.
        
        _, batch = train_loader_iter.__next__()
        inputs_ll, inputs_wl, targets, valid, meta = batch

        input_var = torch.autograd.Variable(inputs_ll.cuda())
        input_var_wl = torch.autograd.Variable(inputs_wl.cuda())

        plot_images_to_wandb([input_var[0],input_var_wl[0]], "Comparison")

        target15, target11, target9, target7 = targets
        refine_target_var = torch.autograd.Variable(target7.cuda(non_blocking =True))
        valid_var = torch.autograd.Variable(valid.cuda(non_blocking =True))

        l0, l1, l2, l3, l4, global_outputs, refine_output = model(input_var, ll_idx * torch.ones(input_var.shape[0], dtype=torch.long).cuda())
        w0, w1, w2, w3, w4, global_outputs_wl, refine_output_wl = model(input_var_wl, wl_idx * torch.ones(input_var_wl.shape[0], dtype=torch.long).cuda())

        low_features = {'layer0':l0, 'layer1':l1, 'layer2':l2, 'layer3':l3, 'layer4':l4}
        well_features = {'layer0':w0, 'layer1':w1, 'layer2':w2,'layer3':w3, 'layer4':w4}
        lupi_weights = {'layer0':0.2, 'layer1':0.2, 'layer2':0.2, 'layer3':0.2, 'layer4':0.2}


        for idx, layer in enumerate(lupi_weights):
            well_feature = well_features[layer]
            low_feature = low_features[layer]
            layer_lupi_loss = 0

            for batch_idx in range(low_feature.size(0)):
                low_gram = gram_matrix(low_feature[batch_idx])
                well_gram = gram_matrix(well_feature[batch_idx])
                n,d,h,w = well_feature.size()
                layer_lupi_loss += lupi_weights[layer]*lupi_criterion(well_gram.detach(), low_gram)/(h*h*w*w) 
            
            loss_lupi += layer_lupi_loss / 4.


        # if sup == 'all':
        for global_output, label in zip(global_outputs, targets):
            num_points = global_output.size()[1]
            global_label = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss = criterion1(global_output, torch.autograd.Variable(global_label.cuda(non_blocking =True))) / 2.0
            loss += global_loss
            global_loss_record += global_loss.data.item()
        refine_loss = criterion2(refine_output, refine_target_var)
        refine_loss = refine_loss.mean(dim=3).mean(dim=2)
        refine_loss *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss = ohkm(refine_loss, 8)
        loss += refine_loss
        refine_loss_record = refine_loss.data.item()

        for global_output_wl, label in zip(global_outputs_wl, targets):
            num_points = global_output_wl.size()[1]
            global_label_wl = label * (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
            global_loss_wl = criterion1(global_output_wl, torch.autograd.Variable(global_label_wl.cuda(non_blocking =True))) / 2.0
            loss += global_loss_wl
            global_loss_record_wl += global_loss_wl.data.item()

        refine_loss_wl = criterion2(refine_output_wl, refine_target_var)
        refine_loss_wl = refine_loss_wl.mean(dim=3).mean(dim=2)
        refine_loss_wl *= (valid_var > 0.1).type(torch.cuda.FloatTensor)
        refine_loss_wl = ohkm(refine_loss_wl, 8)
        loss += refine_loss_wl
        refine_loss_record_wl = refine_loss_wl.data.item()
        

        lupi_w = 0.0001
        loss += lupi_w*loss_lupi

        wandb.log({"loss": loss.data.item()})
        wandb.log({"lupi_loss": lupi_w*loss_lupi.data.cpu().numpy() })

        # # record loss
        losses.update(loss.data.item(), inputs_ll.size(0))
        
        wandb.log({"global_loss_record_wl":global_loss_record_wl})
        wandb.log({"global_loss_record_ll":global_loss_record})
        wandb.log({"refine_loss_record_wl":refine_loss_record_wl})
        wandb.log({"refine_loss_record_ll":refine_loss_record})

        optimizer.zero_grad()
        loss.backward()


        optimizer.step()

        if(i%100==0 and i!=0):
            print('iteration {} | loss: {}, global loss: {}, refine loss: {}, avg loss: {}'
                .format(i, loss.data.item(), global_loss_record, 
                    refine_loss_record, losses.avg)) 

    return losses.avg

NAME = 'Ours'
PATH_NAME = os.path.join('./checkpoint/', NAME)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CPN Training')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 12)')
    parser.add_argument('-g', '--num_gpus', default=1, type=int, metavar='N',
                        help='number of GPU to use (default: 1)')    
    parser.add_argument('--epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run (default: 32)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-c', '--checkpoint', default=NAME, type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    parser.add_argument('--file-name', default=NAME, type=str)

    main(parser.parse_args())
