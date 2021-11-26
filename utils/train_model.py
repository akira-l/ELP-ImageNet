#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime
import random
import gc

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
#from torchvision.utils import make_grid, save_image

import torch.distributed as dist
from utils.utils import LossRecord, clip_gradient, weights_normal_init
from utils.eval_model import eval_turn
from models.focal_loss import FocalLoss
from utils.Asoftmax_loss import AngleLoss
from utils.swap_op import SwapMapping
from utils.adjust_lr import adjust_learning_rate
from models.entropy_2d import Entropy
from models.align_loss import align_loss
#from utils.logger import Logger
from utils.NCEAverage import NCEAverage 
from utils.LinearAverage import LinearAverage 

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model_recieve,
          epoch_num,
          start_epoch,
          optimizer_recieve,
          scheduler_recieve,
          data_loader,
          save_dir,
          data_ver='all',
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):


    if isinstance(model_recieve, dict):
        model = model_recieve['base']
        avg_bank = model_recieve['align']
        as_model = model_recieve['bank']


    if isinstance(optimizer_recieve, dict):
        optimizer = optimizer_recieve['common']
        filter_optim = optimizer_recieve['filter']

    if isinstance(scheduler_recieve, dict):
        exp_lr_scheduler = scheduler_recieve['common']
        filter_exp_lr_scheduler = scheduler_recieve['filter']

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []
    max_record = []
    dis_resave_dir = '/data01/liangyzh/mhem_distribution_resave'

    train_batch_size = data_loader['train'].batch_size
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    #logger = Logger('./tb_logs')

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()

    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_nonreduce_celoss = nn.CrossEntropyLoss(reduction='none')
    get_kld_loss = nn.KLDivLoss(reduction='batchmean')
    get_l1_loss = nn.L1Loss()
    get_l2_loss = nn.MSELoss()
    get_smooth_l1_loss = nn.SmoothL1Loss()
    get_sim_loss_noreduce = nn.CosineEmbeddingLoss(reduction='none')
    get_sim_loss = nn.CosineEmbeddingLoss()
    get_entropy = Entropy()
    #get_swap_op = SwapMapping(9, [7, 7])
    get_sim = nn.CosineSimilarity()

    #writer = SummaryWriter('./re_tb_logs')

    gate_thresh = 0
    epoch_gate = 200
    ep_th = 60

    #nce_k = 4096 
    #if nce_k:
    #    lemniscate = NCEAverage(args.low_dim, ndata, args.nce_k, args.nce_t, args.nce_m)
    #else:
    #    lemniscate = LinearAverage(args.low_dim, ndata, args.nce_t, args.nce_m)

    #lemniscate.cuda() 

    sim_training = True# False
    for epoch in range(start_epoch,epoch_num-1):
        gather_score = {'name': [],
                        'tmp_score': [],
                        'main_score': []}

        #optimizer = adjust_learning_rate(optimizer, epoch)
        model.train(True)

        anno_gather_dict = {}
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            as_model.train(True)

            inputs, labels, img_names = data
            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            optimizer.zero_grad()
            filter_optim.zero_grad()

            outputs, cls_feat, avg_feat, top_avg_feat = model(inputs, labels, img_names)

            #bank_feat = avg_bank.module.return_bank_feat()
            bank_feat = avg_bank.feat_bank.cuda()
            pred_val, pred_ind = outputs.max(1)
            #print('bank_feat', bank_feat)
            #print('bank_feat shape', bank_feat.shape)
            #print('labels', labels)
            #print('labels shape', labels.shape)
            gather_mem = bank_feat[labels.detach()].detach()
            mem_out = as_model(gather_mem)
            #mem_out = as_model(avg_feat)
            avg_bank.proc(outputs.detach().cpu(), labels.detach().cpu(), avg_feat.detach().cpu())
            #avg_bank.module.proc_bank(outputs, labels, avg_feat)

            mix_label = torch.randint(0, Config.numcls, (2, len(labels))).long()  
            mix_label = Variable(mix_label.cuda())
            mix_feat = 0.5 * bank_feat[mix_label[0]] + 0.5 * bank_feat[mix_label[1]] 
            mix_out = as_model(mix_feat.detach()) 
            mix_mem_loss = 0.5 * get_ce_loss(mix_out, mix_label[0]) + 0.5 * get_ce_loss(mix_out, mix_label[1])  
            #mix_mem_loss = 0.5 * mix_mem_loss 

            ce_loss = get_ce_loss(outputs, labels)

            #tmp_update_bank = 0.9 * bank_feat[labels] + 0.1 * avg_feat
            #tmp_update_bank = Config.mem_m * bank_feat[labels] + (1 - Config.mem_m) * avg_feat
            tmp_update_bank = avg_feat

            tmp_out = as_model.module.sp_Acls(tmp_update_bank.detach())
            tmp_score = F.softmax(tmp_out.detach(), 1)
            main_score = F.softmax(outputs, 1)

            if Config.train_ver == 'sum': 
                alpha = Config.alpha  
                sum_score = alpha * main_score + (1 - alpha) * tmp_score
                div_score = (main_score - 1 + 2*tmp_score) / sum_score 
                div_score = div_score.detach() 

            if Config.train_ver == 'mul': 
                mul_score = main_score * tmp_score
                tmp_sub_score = mul_score - main_score + tmp_score   
                div_score = (mul_score - 1 + tmp_score) / mul_score 
                div_score = div_score.detach() 
 
            sel_mask = torch.FloatTensor(len(tmp_score), Config.numcls).zero_().cuda()
            sel_mask.scatter_(1, labels.unsqueeze(1), 1.0)
            sel_mask.cuda()

            sel_prob = (div_score * sel_mask).sum(1).view(-1, 1)
            sel_prob = torch.clamp(sel_prob, 1e-8, 1 - 1e-8)

            gamma = 2
            mem_focal = - torch.pow(1 - sel_prob, gamma) * main_score.log()
            mem_focal = mem_focal.mean()


            mem_loss = get_ce_loss(mem_out, labels)
            if torch.isnan(mem_focal): 
                mem_focal = 0*ce_loss 
            

            #mem_focal = get_smooth_l1_loss(main_score, tmp_score)
            #mem_focal = get_l2_loss(main_score, tmp_score)
            loss = ce_loss + mem_loss + mem_focal #+ mix_mem_loss # + err_mem_loss + all_mem_coss
            #loss = ce_loss + mem_loss #+ mem_focal # + err_mem_loss + all_mem_coss
            #loss = ce_loss + mem_focal

            loss.backward()
            #torch.nn.utils.clip_grad_norm_(as_model.module.parameters(), 0.4)
            #torch.nn.utils.clip_grad_norm_(model.module.parameters(), 0.4)

            optimizer.step()
            filter_optim.step()

            avg_bank.update_bank()
            print('step: {:-8d} / {:d} loss=ce+entropy+smooth_l1_map: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} + {:6.4f} '.format(step, train_epoch_step,
                                                                                                                   loss.detach().item(),
                                                                                                                   ce_loss.detach().item(),
                                                                                                                   mem_loss.detach().item(),
                                                                                                                   mix_mem_loss.detach().item(),
                                                                                                                   mem_focal.detach().item(),
                                                                                                                   ))
            train_loss_recorder.update(loss.detach().item())

            torch.cuda.synchronize()
            #torch.cuda.empty_cache()

            # evaluation & save
            if step % checkpoint == 0:
                model_dict = {}
                model_dict['base'] = model
                model_dict['align'] = avg_bank
                model_dict['bank'] = as_model
                rec_loss = []
                print(32*'-')
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()))
                print('current lr:%s' % exp_lr_scheduler.get_lr())
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(model_dict, data_loader['trainval'], 'trainval', epoch, Config)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(model_dict, data_loader['val'], 'val', epoch, Config)


                save_path = os.path.join(save_dir, 'weights__base__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                as_save_path = os.path.join(save_dir, 'weights_as_model_base-pick50__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                mem_save_path = os.path.join(save_dir, 'avg_memory__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                torch.cuda.synchronize()
                #torch.save(model.state_dict(), save_path)
                #if epoch %10 == 0:# and val_acc1 > max(max_record):
                #    torch.save(model.state_dict(), save_path)
                #    torch.save(as_model.state_dict(), as_save_path)
                #    torch.save(avg_bank.feat_bank, mem_save_path)

                print('saved model to %s' % (save_path))
                max_record.append(val_acc1)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint__base__weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                #torch.save(model.state_dict(), save_path)

                torch.cuda.empty_cache()
            #acc = kNN(0, model, lemniscate, data_loader['train'], data_loader['val'], 1000, nce_t, 1)
            
        as_save_path = os.path.join(save_dir, 'weights_as_model_base__%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
        torch.save(as_model.state_dict(), as_save_path) 
        if epoch > 3: 
            raise Exception('done') 

        #torch.save(gather_score, 'gather_score_' + str(epoch) +  '.pt')
        exp_lr_scheduler.step(epoch)
        filter_exp_lr_scheduler.step(epoch)

        if epoch % 2 == 0:
            weights_normal_init(as_model)



        gc.collect()




