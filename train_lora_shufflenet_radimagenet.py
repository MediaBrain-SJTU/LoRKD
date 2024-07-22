import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import copy
import threading
import random
import multiprocessing
from torch.utils.data import Dataset, DataLoader, Sampler

from losses import SoftTarget
from data.radimagenet import RadData

from network.shuffle_adapter import AdapterWrapperShuffleNet
from peft.lora_fast_shufflenet import LoraConv2d, MultiLoraConv2d
import time
from network.shufflenet_v2_lora import shufflenet
from collections import OrderedDict


def main():

    num_task = 11
    num_classes = [6,28,2,13,18,14,9,25,26,10,14]
    total_num_classes = 165

    max_epochs = 100
    batch_size = 1024
    test_batch_size = 512
    multi_gpu = True

    # model
    adapter_class = MultiLoraConv2d
    resnet = shufflenet(pretrained=False)

    student_model = nn.ModuleDict(
        {
            'CKN': AdapterWrapperShuffleNet(resnet, adapter_class, num_task=num_task, gamma=4, lora_alpha=16), # gamma=8, lora_alpha=16
            'neck': nn.AdaptiveAvgPool2d((1, 1)),
            'head_task': nn.ModuleList([nn.Linear(1024, num_classes[i]) for i in range(num_task)]),
            'head': nn.Linear(1024, total_num_classes),
        }
    )


    if multi_gpu:
        for each_key in student_model.keys():
            if isinstance(student_model[each_key], nn.ModuleList):
                student_model[each_key] = nn.ModuleList([nn.DataParallel(each_module) for each_module in student_model[each_key]])
            else:
                student_model[each_key] = nn.DataParallel(student_model[each_key])

    student_model = student_model.cuda()


    teacher_model = nn.ModuleDict(
        {
            'backbone': radresnet50(model_path='RadImageNet-ResNet50_notop.pth'),
            'neck': nn.AdaptiveAvgPool2d((1, 1)),
        }
    )


    if multi_gpu:
        for each_key in teacher_model.keys():
            teacher_model[each_key] = nn.DataParallel(teacher_model[each_key])


    feat_channels_student = [1024]
    feat_channels_teacher = [2048]
    feat_fcs = []
    for i in range(len(feat_channels_student)):
        feat_fcs.append(nn.Sequential(
            nn.Linear(
                feat_channels_teacher[i], feat_channels_student[i]),
        )
        )
    feat_fcs = nn.ModuleList(feat_fcs).cuda()

    teacher_model = teacher_model.cuda()
    teacher_model.eval()
    teacher_model['backbone'].eval()
    if 'head' in teacher_model.keys():
        teacher_model['head'].eval()


    # loss function
    criterionCls = F.cross_entropy
    criterionKD = SoftTarget(10.0)

    # optimizer
    trainable_list = nn.ModuleList([])
    trainable_list.append(student_model)
    if feat_fcs is not None:
        trainable_list.append(feat_fcs)
    optimizer = torch.optim.SGD(
        trainable_list.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0001,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    print('build optimizer finish!')

    # data
    train_dataset = RadData(split="train")
    test_dateset = RadData(split="test")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dateset, batch_size=test_batch_size, shuffle=False, num_workers=4)


    # train
    tensor_num_classes = torch.tensor(num_classes)
    tensor_num_classes_cumsum = tensor_num_classes.cumsum(dim=0)

    print('start train...')
    for epoch in range(max_epochs):
        start_time = time.time()
        for idx, batch_data in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}")):
            optimizer.zero_grad()
            img, gt_label, task_label = batch_data
            real_bs, real_c = img.shape[0], img.shape[1]
            if img.shape[0] < batch_size:
                continue

            img, gt_label, task_label = img.cuda(), gt_label.cuda(), task_label.cuda()

            with torch.no_grad():
                teacher_feature_before_pool = teacher_model['backbone'](img[:real_bs,:])
                teacher_feature = teacher_model['neck'](teacher_feature_before_pool)
                teacher_feature = teacher_feature.view(teacher_feature.size(0), -1)
                if 'head' in teacher_model.keys():
                    teacher_logit = teacher_model['head'](teacher_feature)
            
            if 'head' not in teacher_model.keys():
                    teacher_feature = feat_fcs[0](teacher_feature)

            multi_student_feature_before_pool = student_model['CKN'](img, task_label)
            multi_student_feature = student_model['neck'](multi_student_feature_before_pool).view(multi_student_feature_before_pool.size(0), -1)

            result = dict(feats=[], mu_vars=[])

            # get logit
            student_logits = student_model['head'](multi_student_feature)

            # ========= loss ========== #
            loss_task = 0.0
            count = 0
            total_num_count = 0
            batch_size = img.shape[0]
            for i in range(num_task):
                task_label_not_one_hot = torch.argmax(task_label, dim=1)
                task_select_mask = task_label_not_one_hot == i
                if torch.sum(task_select_mask) == 0:
                    continue
                logit = student_model['head_task'][i](multi_student_feature[task_select_mask])
                if i>0:
                    label_offset = tensor_num_classes_cumsum[i-1]
                else:
                    label_offset = 0
                count = count + 1
                total_num_count = total_num_count + len(multi_student_feature[task_select_mask])
                loss_task = loss_task + len(multi_student_feature[task_select_mask])*(criterionCls(logit, gt_label.view(-1)[task_select_mask]-label_offset) + 0*criterionKD(multi_student_feature[task_select_mask], teacher_feature.detach()[task_select_mask]))
            loss_task = loss_task/batch_size

            if 'head' in teacher_model.keys():
                loss_kd = criterionKD(student_logits, teacher_logit.detach())
            else:
                loss_kd = criterionKD(multi_student_feature, teacher_feature.detach())

            # Cls loss and infor loss
            loss_cls = criterionCls(student_logits, gt_label.view(-1))

            total_loss = 0.*(loss_kd + loss_cls) + loss_task 
            total_loss.backward()
            optimizer.step()
        scheduler.step()



    # test
    print('start test...')
    student_model.eval()
    correct, total = 0, 0
    correct_list, total_list = torch.tensor([0]*num_task), torch.tensor([0]*num_task)

    for idx, batch_data in enumerate(tqdm(test_dataloader, desc=f"Test")):
        img, gt_label, task_label = batch_data
        if img.shape[0] < test_batch_size:
            continue
        real_bs = img.shape[0]

        img, gt_label, task_label = img.cuda(), gt_label.cuda(), task_label.cuda()
        gt_label = gt_label.view(-1)
        
        multi_student_feature_before_pool = student_model['CKN'](img, task_label)
        multi_student_feature = student_model['neck'](multi_student_feature_before_pool).view(multi_student_feature_before_pool.size(0), -1)

        result = dict(feats=[])

        # get logit
        student_logits = student_model['head'](multi_student_feature) 
        
        for i in range(num_task):
            task_label_not_one_hot = torch.argmax(task_label, dim=1)
            task_select_mask = task_label_not_one_hot == i
            if torch.sum(task_select_mask) == 0:
                continue
            logit = student_model['head_task'][i](multi_student_feature[task_select_mask])
            if i>0:
                label_offset = tensor_num_classes_cumsum[i-1]
            else:
                label_offset = 0

            pred = torch.argmax(logit, dim=1) 

            correct_list[i] = correct_list[i] + torch.sum(pred == (gt_label[task_select_mask]-label_offset)).item()
            total_list[i] = total_list[i] + len(logit)

        pred = torch.argmax(student_logits, dim=1)
        correct += torch.sum(pred == gt_label).item()
        total += len(gt_label)

    print(f"Indomain Evaluating Accuracy: {correct/total: .4f}")
    print(f"Indomain Evaluating each task Accuracy: {correct_list/total_list}")


    torch.save(student_model.state_dict(), 'lora_radimagenet_shufflenet.pth')


if __name__ == "__main__":
    main()
