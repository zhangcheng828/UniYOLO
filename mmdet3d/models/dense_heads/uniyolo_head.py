# Copyright (c) OpenMMLab. All rights reserved.
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)

from mmcv.runner import force_fp32
from mmdet.core.bbox.builder import build_bbox_coder
from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet3d.core import (box3d_multiclass_nms,
                          xywhr2xyxyr)
from mmcv.ops.nms import batched_nms

from mmdet.models.builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class UniYOLOHead(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 eval_type='box_iou',
                 feat_channels=256,
                 stacked_convs=3,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_depth=dict(
                     type='LaplacianAleatoricUncertaintyLoss', 
                     loss_weight=1.0
                 ),
                 loss_dim=dict(type='L1Loss',  loss_weight=1.0),
                 loss_alpha_cls = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_alpha_reg = dict(type='L1Loss', loss_weight=1.0),
                 loss_offset = dict(type='L1Loss',  loss_weight=1.0),
                 bbox_coder=dict(type='MonoYOLOXCoder'),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu')):

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.num_alpha_bins = 12
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)
        self.loss_dim = build_loss(loss_dim)
        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False
        self.loss_depth = build_loss(loss_depth)
        self.loss_alpha_cls = build_loss(loss_alpha_cls)
        self.loss_alpha_reg = build_loss(loss_alpha_reg)
        self.loss_offset = build_loss(loss_offset)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.fp16_enabled = False
        self.pred_bbox2d=True
        self._init_layers()
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.eval_type = eval_type
    
    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_bbox_3d = nn.ModuleList()
        self.multi_level_conv_dir_feat = nn.ModuleList()
        self.multi_level_conv_dir = nn.ModuleList()
        self.multi_level_conv_kpts = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            (conv_cls, conv_reg, conv_obj, conv_bbox_3d, conv_dir_feat, 
                        conv_dir, conv_kpts) = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            self.multi_level_conv_bbox_3d.append(conv_bbox_3d)
            self.multi_level_conv_dir_feat.append(conv_dir_feat)
            self.multi_level_conv_dir.append(conv_dir)
            self.multi_level_conv_kpts.append(conv_kpts)


    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        conv_bbox_3d = nn.Conv2d(self.feat_channels, 7, 1) # offset[2] + depth[2] + dims[3]
        conv_kpts = nn.Conv2d(self.feat_channels, 20, 1)

        conv_dir_feat = nn.Sequential(
            nn.Conv2d(self.feat_channels, self.feat_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.feat_channels),
            nn.ReLU(inplace=True),
        )
        conv_dir = nn.Conv2d(self.feat_channels, 2 * self.num_alpha_bins, kernel_size=1)
        
        return conv_cls, conv_reg, conv_obj, conv_bbox_3d, conv_dir_feat, conv_dir, conv_kpts

    def init_weights(self):
        super(UniYOLOHead, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj, conv_bbox_3d, conv_dir_feat, conv_dir):
        """Forward feature of a single scale level."""

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)

        bbox_3d = conv_bbox_3d(reg_feat)
        dir_feat = conv_dir_feat(reg_feat)

        dir_pred = conv_dir(dir_feat)

        return cls_score, bbox_pred, objectness, bbox_3d, dir_pred

    def forward(self, feats):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        return multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj,
                           self.multi_level_conv_bbox_3d,
                           self.multi_level_conv_dir_feat,
                           self.multi_level_conv_dir)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses', 'bbox_3d_preds', 
                    'dir_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   objectnesses,
                   bbox_3d_preds, 
                   dir_preds, 
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        scale_factors = np.array(
            [img_meta['scale_factor'] for img_meta in img_metas])
        box_type_3d = img_metas[0]['box_type_3d']
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_bbox_3d_preds = [
            bbox_3d_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 7)
            for bbox_3d_pred in bbox_3d_preds
        ]
        flatten_dir_preds = [
            dir_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2 * self.num_alpha_bins)
            for dir_pred in dir_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()

        flatten_bbox_3d_preds = torch.cat(flatten_bbox_3d_preds, dim=1)
        flatten_dir_preds = torch.cat(flatten_dir_preds, dim=1)

        flatten_priors = torch.cat(mlvl_priors)

        flatten_bboxes = self.bbox_coder.decode_bboxes2d(flatten_bbox_preds, flatten_priors)
        flatten_bboxes3d, flatten_sigma = self.bbox_coder.decode_bboxes3d(flatten_bbox_3d_preds, 
                                            flatten_dir_preds, flatten_priors,
                                            img_metas[0]['cam2img'])

        if rescale:
            flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
                scale_factors).unsqueeze(1)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_scores = flatten_cls_scores[img_id]
            score_factor = flatten_objectness[img_id]
            bboxes = flatten_bboxes[img_id]
            bboxes_3d = flatten_bboxes3d[img_id]
            sigma = flatten_sigma[img_id]

            bbox, bbox_3d, scores, labels = self.bboxes_nms(cls_scores, bboxes, 
                                                            bboxes_3d, score_factor, sigma, box_type_3d, cfg)
            result_list.append([
                box_type_3d(bbox_3d, box_dim=7, origin=(0.5, 0.5, 0.5)),
                scores,
                labels,
                bbox,])

        return result_list
            

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses','bbox_3d_preds',
                        'alpha_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             objectnesses,
             bbox_3d_preds,
             alpha_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             gt_centers2d,
             gt_depths,
             img_metas):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        num_imgs = len(img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)

        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_bbox_3d_preds = [
            bbox_3d_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 7)
            for bbox_3d_pred in bbox_3d_preds
        ]
        flatten_alpha_preds = [
            alpha_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2 * self.num_alpha_bins)
            for alpha_pred in alpha_preds
        ]


        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_bbox_3d_preds = torch.cat(flatten_bbox_3d_preds, dim=1)
        flatten_alpha_preds = torch.cat(flatten_alpha_preds, dim=1)

        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self.bbox_coder.decode_bboxes2d(flatten_bbox_preds,flatten_priors)

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, dim_targets, 
         depth_targets, alpha_cls_targets, alpha_offset_targets, offset_target, 
        cam2imgs, num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), 
             gt_bboxes, 
             gt_labels_3d,
             gt_bboxes_3d,
             gt_depths,
             gt_centers2d,
             img_metas,
             )

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        dim_targets = torch.cat(dim_targets, 0)
        depth_targets = torch.cat(depth_targets, 0)
        alpha_cls_targets = torch.cat(alpha_cls_targets, 0)
        alpha_offset_targets = torch.cat(alpha_offset_targets, 0)
        offset_target = torch.cat(offset_target, 0)

        cam2imgs = torch.cat(cam2imgs, 0)

        # alpha cls
        alpha_cls_pred = flatten_alpha_preds[..., :self.num_alpha_bins].view(-1, self.num_alpha_bins)[pos_masks]
        alpha_cls_targets = alpha_cls_targets.type(torch.long)
        alpha_cls_onehot_target = alpha_cls_targets.new_zeros([len(alpha_cls_targets), self.num_alpha_bins]).scatter_(
            dim=1, index=alpha_cls_targets.view(-1, 1), value=1)
        
        # alpha offset
        alpha_offset_pred = flatten_alpha_preds[..., self.num_alpha_bins:].view(-1, self.num_alpha_bins)[pos_masks]
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)

        if pos_masks.sum() > 0:
            loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        loss_alpha_reg = self.loss_alpha_reg(alpha_offset_pred, alpha_offset_targets)

        dim_preds = flatten_bbox_3d_preds[..., 4:].view(-1, 3)[pos_masks]
        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_preds, dim_targets, dim_preds)
        else:
            loss_dim = self.loss_dim(dim_preds, dim_targets)
        
        loss_offset_2d = self.loss_offset(flatten_bbox_3d_preds[..., :2].view(-1, 2)[pos_masks], offset_target)
       
        flatten_depth_preds = flatten_bbox_3d_preds[..., 2:4].view(-1, 2)[pos_masks]
        depth_pred, depth_log_variance = flatten_depth_preds[:, 0:1], flatten_depth_preds[:, 1:2]

        direct_depth = self.bbox_coder.decode_depth(depth_pred)
        loss_depth = self.loss_depth(direct_depth, depth_targets, depth_log_variance)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples

        loss_dict = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox, loss_obj=loss_obj, loss_depth=loss_depth, 
            loss_dim=loss_dim, loss_offset_2d=loss_offset_2d, 
            loss_alpha_cls=loss_alpha_cls, loss_alpha_reg=loss_alpha_reg)

        return loss_dict

    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                           gt_bboxes, gt_labels, gt_bboxes_3d, gt_depths, gt_centers2d,img_meta):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        gt_depths = gt_depths.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            dim_target = decoded_bboxes.new_zeros((0, 3))
            depth_target = decoded_bboxes.new_zeros((0))
            alpha_cls_target = decoded_bboxes.new_zeros((0, 1))
            alpha_offset_target = decoded_bboxes.new_zeros((0, 1))
            offset_target = decoded_bboxes.new_zeros((0, 2))

            cam2img = cls_preds.new_zeros((0, 4, 4))
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, dim_target, depth_target, alpha_cls_target, 
                    alpha_offset_target, offset_target, cam2img, 0)

        gt_bbox_3d = gt_bboxes_3d.tensor.to(decoded_bboxes.dtype)
        gt_bbox_3d[..., 6] = -torch.atan2(
            gt_bbox_3d[..., 0], gt_bbox_3d[..., 2]) + gt_bbox_3d[..., 6]
        dim_target = gt_bbox_3d[:, 3:6].to(gt_bboxes.device)
        depth_target = gt_depths
        alpha_cls_target = decoded_bboxes.new_zeros((num_gts, 1))
        alpha_offset_target = decoded_bboxes.new_zeros((num_gts, 1))

        for i, bbox_3d in enumerate(gt_bbox_3d):
            alpha_cls_target[i], alpha_offset_target[i] = self.bbox_coder.angle2class(bbox_3d[6])
            
        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds

        num_pos_per_img = pos_inds.size(0)
        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)

        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))

        gt_centers2d = gt_centers2d[pos_assigned_gt_inds]
        center_offset_target = cls_preds.new_zeros((num_pos_per_img, 2))
        center_offset_target = self._get_center_offset_target(center_offset_target,
                                                              gt_centers2d,
                                                              priors[pos_inds])

        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1
        dim_target = dim_target[pos_assigned_gt_inds]
        depth_target = depth_target[pos_assigned_gt_inds]
        alpha_cls_target = alpha_cls_target[pos_assigned_gt_inds]
        alpha_offset_target = alpha_offset_target[pos_assigned_gt_inds]

        cam2img = objectness.new_tensor(img_meta['cam2img']).unsqueeze(0).repeat(num_pos_per_img, 1, 1)

        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, dim_target, depth_target, alpha_cls_target, 
                alpha_offset_target, center_offset_target, cam2img, num_pos_per_img)

    def _get_center_offset_target(self, center_offset_target, gt_centers, priors, broad=False):
        """Convert gt bboxes3d to center offset"""
        if broad: priors = priors.unsqueeze(1) 
        center_offset_target[..., :2] = (gt_centers[..., :2] - priors[..., :2]) / priors[..., 2:]
        return center_offset_target


    def bboxes_nms(self, cls_scores, bboxes, batch_bboxes_3d, score_factor, sigma, box_type_3d, cfg):
    
        if self.eval_type == 'box_iou':
            max_scores, labels = torch.max(cls_scores, 1)
            valid_mask = score_factor * max_scores >= cfg.score_thr

            bboxes = bboxes[valid_mask]
            labels = labels[valid_mask]
            batch_bboxes_3d = batch_bboxes_3d[valid_mask]
            scores = max_scores[valid_mask] * score_factor[valid_mask] * sigma[valid_mask]
            if labels.numel() == 0:
                return bboxes, batch_bboxes_3d, scores, labels
            else:
                dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
                return dets, batch_bboxes_3d[keep], scores[keep], labels[keep]
        elif self.eval_type == '3D_iou':
            cfg.use_rotate_nms = True
            cfg.nms_thr = 0.7
            batch_bboxes_3d_for_nms = xywhr2xyxyr(box_type_3d(
                batch_bboxes_3d, box_dim=7, origin=(0.5, 0.5, 0.5)).bev)
            padding = cls_scores.new_zeros(cls_scores.shape[0], 1)
            # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
            # BG cat_id: num_class
            cls_scores = torch.cat([cls_scores, padding], dim=1)
            cls_scores = cls_scores * sigma[:, None]
            results = box3d_multiclass_nms(batch_bboxes_3d, batch_bboxes_3d_for_nms,
                                        cls_scores, 0.2,
                                        30, cfg, mlvl_bboxes2d=bboxes)
            batch_bboxes_3d, scores, labels, bboxes = results
            bboxes = torch.cat([bboxes, scores.reshape(-1, 1)], dim=1)
            return bboxes, batch_bboxes_3d, scores, labels


    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        assert gt_labels is not None
        assert attr_labels is None
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d, gt_labels_3d, centers2d, 
                              depths, img_metas)

        losses = self.loss(*loss_inputs)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError