import time
import datetime
import logging
import torch
torch.autograd.set_detect_anomaly(True)
from apex import amp
from tools.utils import AverageMeter
from losses.aug_loss import domain_recon_loss, domain_tri_loss, domain_align_loss, feat_aug_loss, feat_inner_aug_loss
import torch.nn.functional as F

def train_cal(config, epoch, model, domain_aug, feat_aug, classifier, model_aug, domain_aug_, clothes_classifier, criterion_cla, criterion_pair,
    criterion_clothes, criterion_adv, optimizer, optimizer_cc, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_clo_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    batch_aug_loss = AverageMeter()
    batch_aug_recon_loss = AverageMeter()
    batch_aug_tri_loss = AverageMeter()
    batch_aug_align_loss = AverageMeter()
    batch_aug_feat_tri_loss = AverageMeter()
    corrects = AverageMeter()
    clothes_corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()
    clothes_classifier.train()
    domain_aug.train()
    feat_aug.train()

    end = time.time()
    for batch_idx, (imgs, aug_imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        x = model.module.base(imgs)
        feats = model.module.globalpooling(x).view(x.size(0), -1)
        features = model.module.bn(feats)

        # features = model(imgs)
        outputs = classifier(features)

        model_state_dict = model.module.state_dict()
        model_aug.load_state_dict(model_state_dict)
        ref_feats = ref(model_aug, aug_imgs)
        _, domain_feats = domain_aug(feats)
        domain_features = torch.stack([model.module.bn(i) for i in domain_feats], dim=0)
        pred_clothes = clothes_classifier(features.detach())
        _, preds = torch.max(outputs.data, 1)

        features_aug_inter, features_aug_inner  = feat_aug(features, domain_features, pids) 

        # Update the clothes discriminator
        clothes_loss = criterion_clothes(pred_clothes, clothes_ids)
        if epoch >= config.TRAIN.START_EPOCH_CC:
            optimizer_cc.zero_grad()
            if config.TRAIN.AMP:
                with amp.scale_loss(clothes_loss, optimizer_cc) as scaled_loss:
                    scaled_loss.backward()
            else:
                clothes_loss.backward()
            optimizer_cc.step()

        # Update the backbone
        new_pred_clothes = clothes_classifier(features)
        _, clothes_preds = torch.max(new_pred_clothes.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)
        adv_loss = criterion_adv(new_pred_clothes, clothes_ids, pos_mask)
        aug_recon_loss = domain_recon_loss(domain_feats, feats)
        aug_tri_loss =  domain_tri_loss(domain_features, pids)
        aug_align_loss = domain_align_loss(domain_feats, ref_feats)
        aug_feat_tri_loss =  feat_aug_loss(features_aug_inter, pids)  + feat_aug_loss(features_aug_inner, pids)  
        if epoch >= config.TRAIN.START_EPOCH_ADV:
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + aug_feat_tri_loss + (aug_recon_loss +  aug_tri_loss + aug_align_loss) 
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss + aug_feat_tri_loss + (aug_recon_loss + aug_tri_loss + aug_align_loss) 
        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        clothes_corrects.update(torch.sum(clothes_preds == clothes_ids.data).float()/clothes_ids.size(0), clothes_ids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        batch_clo_loss.update(clothes_loss.item(), clothes_ids.size(0))
        batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        batch_aug_recon_loss.update(aug_recon_loss.item(), pids.size(0))
        batch_aug_tri_loss.update(aug_tri_loss.item(), pids.size(0))
        batch_aug_align_loss.update(aug_align_loss.item(), pids.size(0))
        batch_aug_feat_tri_loss.update(aug_feat_tri_loss.item(), pids.size(0))       
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                  'Time:{batch_time.sum:.1f}s '
                  'Data:{data_time.sum:.1f}s '
                  'ClaLoss:{cla_loss.avg:.4f} '
                  'PairLoss:{pair_loss.avg:.4f} '
                  'CloLoss:{clo_loss.avg:.4f} '
                  'AdvLoss:{adv_loss.avg:.4f} '                 
                  'aug_recon_loss:{aug_recon_loss.avg:.4f} '
                  'aug_tri_loss:{aug_tri_loss.avg:.4f} '
                  'aug_align_loss:{aug_align_loss.avg:.4f} ' 
                  'aug_feat_loss:{aug_feat_tri_loss.avg:.4f} '                                                                        
                  'Acc:{acc.avg:.2%} '
                  'CloAcc:{clo_acc.avg:.2%} '.format(
                   epoch+1, batch_time=batch_time, data_time=data_time, 
                   cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                   clo_loss=batch_clo_loss, adv_loss=batch_adv_loss, 
                   aug_recon_loss = batch_aug_recon_loss, aug_tri_loss = batch_aug_tri_loss, aug_align_loss = batch_aug_align_loss,
                   aug_feat_tri_loss = batch_aug_feat_tri_loss,
                   acc=corrects, clo_acc=clothes_corrects))

def ref(model, img_list):
    feats = []

    with torch.no_grad():
        for imgs in img_list:
            x = model.base(imgs.cuda())
            features = model.globalpooling(x).view(x.size(0), -1)
            feats.append(features)  

    return feats

def train_cal_with_memory(config, epoch, model, classifier, criterion_cla, criterion_pair, 
    criterion_adv, optimizer, trainloader, pid2clothes):
    logger = logging.getLogger('reid.train')
    batch_cla_loss = AverageMeter()
    batch_pair_loss = AverageMeter()
    batch_adv_loss = AverageMeter()
    corrects = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train()
    classifier.train()

    end = time.time()
    for batch_idx, (imgs, pids, camids, clothes_ids) in enumerate(trainloader):
        # Get all positive clothes classes (belonging to the same identity) for each sample
        pos_mask = pid2clothes[pids]
        imgs, pids, clothes_ids, pos_mask = imgs.cuda(), pids.cuda(), clothes_ids.cuda(), pos_mask.float().cuda()
        # Measure data loading time
        data_time.update(time.time() - end)
        # Forward
        features = model(imgs)
        outputs = classifier(features)
        _, preds = torch.max(outputs.data, 1)

        # Compute loss
        cla_loss = criterion_cla(outputs, pids)
        pair_loss = criterion_pair(features, pids)

        if epoch >= config.TRAIN.START_EPOCH_ADV:
            adv_loss = criterion_adv(features, clothes_ids, pos_mask)
            loss = cla_loss + adv_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss   
        else:
            loss = cla_loss + config.LOSS.PAIR_LOSS_WEIGHT * pair_loss  

        optimizer.zero_grad()
        if config.TRAIN.AMP:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # statistics
        corrects.update(torch.sum(preds == pids.data).float()/pids.size(0), pids.size(0))
        batch_cla_loss.update(cla_loss.item(), pids.size(0))
        batch_pair_loss.update(pair_loss.item(), pids.size(0))
        if epoch >= config.TRAIN.START_EPOCH_ADV: 
            batch_adv_loss.update(adv_loss.item(), clothes_ids.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Epoch{0} '
                'Time:{batch_time.sum:.1f}s '
                'Data:{data_time.sum:.1f}s '
                'ClaLoss:{cla_loss.avg:.4f} '
                'PairLoss:{pair_loss.avg:.4f} '
                'AdvLoss:{adv_loss.avg:.4f} '
                'Acc:{acc.avg:.2%} '.format(
                epoch+1, batch_time=batch_time, data_time=data_time, 
                cla_loss=batch_cla_loss, pair_loss=batch_pair_loss, 
                adv_loss=batch_adv_loss, acc=corrects))