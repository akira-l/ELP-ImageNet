#coding=utf-8
import os,time,datetime
import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from sklearn.manifold import TSNE
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib.pyplot import plot, savefig

from transforms import transforms
from models.LoadModel import MainModel
#from models.primitive_dcl import MainModel
from models.avg_mem import AvgMem
from models.cosine_model import CosSoftmaxModule
from utils.save4submit import Submit_result
from dataset.dataset_DCL import collate_fn4train, collate_fn4test, collate_fn4val, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc
from utils.NCEAverage import NCEAverage 
from utils.LinearAverage import LinearAverage 

import pdb

#os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power
    
    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1./self.power)
        out = x.div(norm)
        return out


def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset',
                        default='imagenet', type=str)
    parser.add_argument('--backbone', dest='backbone',
                        default='resnet50', type=str)
    parser.add_argument('--b', dest='batch_size',
                        default=1024, type=int)
    parser.add_argument('--k', dest='K_val',
                        default=200, type=int)
    parser.add_argument('--tb', dest='train_batch',
                        default=512, type=int)
    parser.add_argument('--ignore_pretrained', dest='ignore_pretrained',
                        action='store_true')
    parser.add_argument('--train_ver', dest='train_ver',
                        default='sum', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        default=32, type=int)
    parser.add_argument('--ver', dest='version',
                        default='val', type=str)
    parser.add_argument('--save', dest='resume',
                        #default='./CUB_base_87.5/weights_264_187_0.8550_0.9465.pth',
                        default=None,
                        type=str)
    parser.add_argument('--size', dest='resize_resolution',
                        default=256, type=int)
    parser.add_argument('--crop', dest='crop_resolution',
                        default=224, type=int)
    #parser.add_argument('--swap_num', dest='swap_num',
    #                    default=7, type=int)
    parser.add_argument('--bad_case',dest='bad_case', action='store_true')
    parser.add_argument('--ss', dest='save_suffix',
                        default=None, type=str)
    parser.add_argument('--score_dir', dest='score_dir',
                        #default='./emsamble_scores',
                        type=str)
    parser.add_argument('--eval_analysis', dest='analysis', action='store_true')
    parser.add_argument('--submit', dest='submit',
                        action='store_true')
    parser.add_argument('--acc_report', dest='acc_report',
                        action='store_true')
    parser.add_argument('--tencroped', dest='tencroped',
                        action='store_true')
    parser.add_argument('--ensamble', dest='ensamble',
                        action='store_true')
    parser.add_argument('--buff2', dest='buff_drop2',
                        action='store_true')
    parser.add_argument('--alpha', dest='alpha',
                        default=0.5, type=float)
    parser.add_argument('--tau', dest='tau',
                        default=0.1, type=float)
    parser.add_argument('--swap_num', default=[7, 7],
                    nargs=2, metavar=('swap1', 'swap2'),
                    type=int, help='specify a range')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.submit:
        args.version = 'test'
        if args.save_suffix == '':
            raise Exception('**** miss --ss save suffix is needed. ')

    #Config = LoadConfig(args, args.version)
    Config = LoadConfig(args, 'val')
    transformers = load_data_transformers(args.resize_resolution, args.crop_resolution, args.swap_num)



    data_set = dataset(Config,\
                       anno=Config.val_anno, #if args.version == 'val' else Config.test_anno ,\
                       unswap=transformers["None"],\
                       swap=transformers["None"],\
                       #totensor=transformers["Tencrop"] if args.tencroped else transformers['test_totensor'],\
                       totensor = transformers["val_totensor"],\
                       test=True)

    dataloader = torch.utils.data.DataLoader(data_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             drop_last=False,
                                             collate_fn=collate_fn4test)

    setattr(dataloader, 'total_item_len', len(data_set))


    train_set = dataset(Config = Config,\
                        anno = Config.train_anno,\
                        unswap = transformers["None"],\
                        swap = transformers["None"],\
                        totensor = transformers["val_totensor"],\
                        test = True)

    retrain_dataloader = torch.utils.data.DataLoader(train_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             drop_last=False,
                                             collate_fn=collate_fn4test)

    setattr(retrain_dataloader, 'total_item_len', len(train_set))




    save_result = Submit_result(args.dataset)
    cudnn.benchmark = True

    #resume = args.resume
    #resume = "./net_model/_5312_herb/weights_86_475_0.8466_0.9448.pth"
    #resume = "./net_model/buff_weights_3_2271_0.8835_0.9621.pth"
    #resume = "weights_7_1967_0.8765_0.9542.pth"
    #resume = './weights_12_563_0.8892_0.9581.pth'
    #resume = '/home/liang/DCL2/submitf-4.22/net_model/4-22-3_butterfly/weights-10-3999-[0.8889].pth'
    #resume = './net_model/butterfly_all_5514/_5514_butterfly/weights_17_6207_0.9334_0.9877.pth'
    #resume = './focal_weights_9_3422_0.8863_0.9560.pth'
    #resume = './net_model/buffall_51414_butterfly/weights_15_3967_0.9539_0.9846.pth'
    #main_resume = '../mem_only_resume_checkpoints/weights__base-pick50__229_187_0.8687_0.9518.pth'
    #as_resume = './bak_vis_cp/_112211_imagenet/weights_as_model_base__2_2502_0.7643_0.9007.pth'
    as_resume = './bak_vis_cp/_112217_imagenet/weights_as_model_base__2_2502_0.7643_0.9003.pth'

    #main_resume = './net_model/_102212_imagenet/weights__base__19_2502_0.7547_0.8958.pth' 
    #main_resume = './net_model/_102212_imagenet/weights__base__83_2502_0.7661_0.9005.pth' 
    #main_resume = './net_model/_102212_imagenet/weights__base__19_2502_0.7547_0.8958.pth' 
    main_resume = './net_model/_102212_imagenet/weights__base__83_2502_0.7661_0.9005.pth' 

    model = MainModel(Config)
    model_dict=model.state_dict()
    pretrained_dict=torch.load(main_resume)
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    #avg_bank = AvgMem(Config, 0.9)

    as_model = CosSoftmaxModule(Config, ver='con')
    as_model_dict = as_model.state_dict() 
    as_pretrained_dict=torch.load(as_resume)
    as_pretrained_dict = {k[7:]: v for k, v in as_pretrained_dict.items() if k[7:] in as_model_dict}
    as_model_dict.update(as_pretrained_dict)
    as_model.load_state_dict(as_model_dict)


    model.cuda()
    model = nn.DataParallel(model)

    as_model.cuda()
    as_model = nn.DataParallel(as_model)
    #nce_k = 4096 
    #if nce_k:
    #    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
    #else:
    #    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    criterion = CrossEntropyLoss()
    model.train(False)
    with torch.no_grad():
        val_corrects1 = 0
        val_corrects2 = 0
        val_corrects3 = 0

        all_val_corrects1 = 0
        all_val_corrects2 = 0
        all_val_corrects3 = 0

        as_val_corrects1 = 0
        as_val_corrects2 = 0
        as_val_corrects3 = 0

        val_size = ceil(len(data_set) / dataloader.batch_size)
        result_gather = {}
        acc1_bad_list = []
        acc2_bad_list = []
        acc3_bad_list = []
        gather_score = []
        gather_name = []

        gt_count_tensor = torch.zeros(1000) 
        pred_count_tensor = torch.zeros(1000) 

        gather_pred = {}
        gather_gt = {}
        gather_as_pred = {}
        gather_as_gt = {}
        correct_name_gather = [] 
        count_bar = tqdm(total=dataloader.__len__())


        top1 = 0. 
        top5 = 0.  
        K = args.K_val 
        C = 1000 
        nce_t = args.tau#0.07 
        total = 0.  
        batchSize = args.batch_size
        retrieval_one_hot = torch.zeros(K, C).cuda()

        trainLabels = None 

        gather_err_handle = open('bem_err_score.txt', 'a')   
        err_name_list = torch.load('bem-base_err_name.pt') 

        for batch_cnt_val, data_val in enumerate(dataloader):
            count_bar.update(1)
            inputs, labels, img_name = data_val
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            # forward
            outputs, cls_feat, avg_feat, top_avg_feat = model(inputs)
            as_outputs = as_model(avg_feat) 

            torch.cuda.synchronize()
            avg_norm_feat = avg_feat.pow(2).sum(1, keepdim=True).pow(1./2) 
            avg_out_feat = avg_feat.div(avg_norm_feat) 

            #dist_gather = [] 
            #if trainLabels is None: 
            #    label_gather = [] 
            #train_count_bar = tqdm(total=retrain_dataloader.__len__())
            #for train_batch_cnt, train_data in enumerate(retrain_dataloader):
            #    train_count_bar.update(1) 
            #    train_inputs, train_labels, train_img_name = train_data
            #    train_inputs = Variable(train_inputs.cuda())

            #    train_outputs, train_cls_feat, train_avg_feat, train_top_avg_feat = model(train_inputs)


            #    torch.cuda.synchronize()

            #    train_avg_norm_feat = train_avg_feat.pow(2).sum(1, keepdim=True).pow(1./2) 
            #    train_avg_out_feat = train_avg_feat.div(train_avg_norm_feat) 

            #    train_sub_dist = torch.mm(avg_out_feat, train_avg_out_feat.t()) 

            #    dist_gather.append(train_sub_dist.detach().cpu()) 
            #    if trainLabels is None: 
            #        label_gather.extend(train_labels)


            #train_count_bar.close() 
            #dist = torch.cat(dist_gather, 1) 

            #if trainLabels is None: 
            #    trainLabels = torch.LongTensor(label_gather).cuda()  
            #batchSize = inputs.size(0) 
            #yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            #candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            #retrieval = torch.gather(candidates, 1, yi.cuda())

            #retrieval_one_hot.resize_(batchSize * K, C).zero_()
            #retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            #yd_transform = yd.clone().div_(nce_t).exp_().cuda() 

            #probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , C), yd_transform.view(batchSize, -1, 1)), 1)
            #_, predictions = probs.sort(1, True)

            ## Find which predictions match the target
            #correct = predictions.eq(labels.data.view(-1,1))

            #top1 = top1 + correct.narrow(1,0,1).sum().item()
            #top5 = top5 + correct.narrow(1,0,5).sum().item()

            #total += labels.size(0)

            #testsize = dataloader.dataset.__len__()
            #print('Test [{}/{}]\t'
            #      'Top1: {:.2f}  Top5: {:.2f}'.format(
            #      total, testsize, top1*100./total, top5*100./total))


            main_out_score = torch.softmax(outputs, 1) 
            main_out_score_val, _ = main_out_score.max(1)
            as_out_score = torch.softmax(as_outputs, 1) 
            as_out_score_val, _ = as_out_score.max(1)

            top3_val, top3_pos = torch.topk(outputs, 3)
            for eval_flag in torch.nonzero(top3_pos[:, 0] == labels).squeeze(1):
                if img_name[eval_flag] in err_name_list: 
                    #gather_err_handle[img_name[eval_flag]] = [main_out_score_val[eval_flag].item(), as_out_score_val[eval_flag].item()] #.write(err_cont) 
                    err_cont = img_name[eval_flag] + ' ' + str(main_out_score[eval_flag, :][labels[eval_flag]].item()) + ' ' + str(as_out_score[eval_flag, :][labels[eval_flag]].item()) + '\n' 
                    gather_err_handle.write(err_cont) 

                #err_cont = img_name[eval_flag] + ' ' + str(main_out_score_ind[eval_flag].item()) + ' ' + str(as_out_score_ind[eval_flag].item()) + '\n' 
                #gather_err_handle[img_name[eval_flag]] = [main_out_score_val[eval_flag].item(), as_out_score_val[eval_flag].item()] #.write(err_cont) 
                

            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            val_corrects1 += batch_corrects1
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()
            val_corrects2 += (batch_corrects2 + batch_corrects1)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()
            val_corrects3 += (batch_corrects3 + batch_corrects2 + batch_corrects1)


    val_acc1 = val_corrects1 / len(data_set)
    val_acc2 = val_corrects2 / len(data_set)
    val_acc3 = val_corrects3 / len(data_set)

    #sub_acc = pred_count_tensor / gt_count_tensor 
    ##torch.save(sub_acc, 'res50-7547_sub_acc_tensor.pt') 
    #torch.save(correct_name_gather, 'res50-7547_correct_name_gather.pt') 
    #print('knn: @', K, ' / ', nce_t, ' : ', top1*100./total)

    val_version = 'test'
    print('--'*30)
    print('noraml eval: ||%s-acc@1: %.4f %s-acc@2: %.4f %s-acc@3: %.4f ' % ( val_version, val_acc1, val_version, val_acc2, val_version, val_acc3))
    print('--' * 30)

    count_bar.close()

    gather_err_handle.close() 
    #torch.save(gather_err_handle, 'bem-base_err_dict.pt') 








