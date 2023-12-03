import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import alexnet, vgg16
from torchvision.ops import roi_pool


class WSPD(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = vgg16(pretrained=False)
        self.roi_output_size = (7, 7)
        self.features = self.base.features[:-1]  # 舍弃最后池化层，只要最后一层卷积+ReLU的结果
        self.fcs = self.base.classifier[:-1]
        self.conv = nn.Conv2d(512, 3, 3, 1,1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_d = nn.Linear(4096, 3)

    def forward(self, batch_imgs, batch_boxes, istrain):
        batch_boxes = [batch_boxes[0]]  # 取出ssw数据，将其转为ssw的list
        out = self.features(batch_imgs)  # 提取特征
        out = roi_pool(out, batch_boxes, self.roi_output_size,
                       1.0 / 16)  # ssw中不同大小proposal归一化

        # 检测模块
        det_out = out.view(len(batch_boxes[0]), -1)  # 将ssw中的proposal拉成向量
        det_out = self.fcs(det_out)  # [27,4096],完成框架中的
        det_scores = F.softmax(self.fc_d(det_out), dim=0)  # 按proposal进行检测打分，列和为1

        # 分类模块
        cls_out = self.conv(out)
        cls_out = self.gap(cls_out)
        cls_scores =F.softmax(cls_out.view(cls_out.size(0), -1), dim=1)

        combined_scores = cls_scores * det_scores  # 对应元素点乘，计算图像得分
        if istrain:
            return combined_scores
        else:
            return det_scores, cls_scores

    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=0)
        image_level_scores = torch.clamp(image_level_scores, min=0.0,
                                         max=1.0)
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss
