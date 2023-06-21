import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.gaussian_target import (gaussian_radius, gen_gaussian_target)
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                                transpose_and_gather_feat)
from mmdet3d.ops.attentive_norm import AttnBatchNorm2d

INF = 1e8
EPS = 1e-12
PI = np.pi


@HEADS.register_module()
class SmokeResizeHead(nn.Module):
    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 bbox3d_code_size=7,
                 num_alpha_bins=12,
                 max_objs=30,
                 vector_regression_level=1,
                 loss_center_heatmap=None,
                 loss_offset=None,
                 loss_dim=None,
                 loss_depth=None,
                 loss_alpha_cls=None,
                 loss_alpha_reg=None,
                 use_AN=True,
                 num_AN_affine=10,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(SmokeResizeHead, self).__init__()
        assert bbox3d_code_size >= 7
        self.num_classes = num_classes
        self.bbox_code_size = bbox3d_code_size
        self.max_objs = 100
        self.num_alpha_bins = num_alpha_bins
        self.vector_regression_level = vector_regression_level

        self.use_AN = use_AN
        self.num_AN_affine = num_AN_affine
        self.norm = AttnBatchNorm2d if use_AN else nn.BatchNorm2d

        self.heatmap_head = self._build_head(in_channel, feat_channel, num_classes)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.dim_head = self._build_head(in_channel, feat_channel, 3)
        self.depth_head = self._build_head(in_channel, feat_channel, 2)
        self._build_dir_head(in_channel, feat_channel)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_offset = build_loss(loss_offset)
        self.loss_dim = build_loss(loss_dim)
        if 'Aware' in loss_dim['type']:
            self.dim_aware_in_loss = True
        else:
            self.dim_aware_in_loss = False
        self.loss_depth = build_loss(loss_depth)
        self.loss_alpha_cls = build_loss(loss_alpha_cls)
        self.loss_alpha_reg = build_loss(loss_alpha_reg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

    def _build_head(self, in_channel, feat_channel, out_channel):
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def _build_dir_head(self, in_channel, feat_channel):
        self.dir_feat = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            self._get_norm_layer(feat_channel),
            nn.ReLU(inplace=True),
        )
        self.dir_cls = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))
        self.dir_reg = nn.Sequential(nn.Conv2d(feat_channel, self.num_alpha_bins, kernel_size=1))

    def _get_norm_layer(self, feat_channel):
        return self.norm(feat_channel, momentum=0.03, eps=0.001) if not self.use_AN else \
            self.norm(feat_channel, self.num_AN_affine, momentum=0.03, eps=0.001)

    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)  # -2.19
        for head in [self.offset_head, self.depth_head,
                     self.dim_head, self.dir_feat,
                     self.dir_cls, self.dir_reg]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        center_heatmap_pred = torch.clamp(center_heatmap_pred, min=1e-4, max=1 - 1e-4)

        offset_pred = self.offset_head(feat)

        dim_pred = self.dim_head(feat)
        depth_pred = self.depth_head(feat)
        depth_pred[:, 0, :, :] = 1. / (depth_pred[:, 0, :, :].sigmoid() + EPS) - 1

        alpha_feat = self.dir_feat(feat)
        alpha_cls_pred = self.dir_cls(alpha_feat)
        alpha_offset_pred = self.dir_reg(alpha_feat)
        return center_heatmap_pred, offset_pred, \
               dim_pred, alpha_cls_pred, alpha_offset_pred, depth_pred

    @force_fp32(apply_to=('center_heatmap_preds',  'offset_preds', 
                          'dim_preds', 'alpha_cls_preds',
                          'alpha_offset_preds', 'depth_preds'))
    def loss(self,
             center_heatmap_preds,
             offset_preds,
             dim_preds,
             alpha_cls_preds,
             alpha_offset_preds,
             depth_preds,
             gt_bboxes,
             gt_labels,
             gt_bboxes_3d,
             gt_labels_3d,
             centers2d,
             depths,
             img_metas,
             attr_labels=None,
             proposal_cfg=None,
             gt_bboxes_ignore=None):

        assert len(center_heatmap_preds) == len(offset_preds) \
               == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1

        center_heatmap_pred = center_heatmap_preds[0]
        offset_pred = offset_preds[0]

        dim_pred = dim_preds[0]
        alpha_cls_pred = alpha_cls_preds[0]
        alpha_offset_pred = alpha_offset_preds[0]
        depth_pred = depth_preds[0]

        batch_size = center_heatmap_pred.shape[0]

        target_result = self.get_targets(gt_bboxes, gt_labels,
                                         gt_bboxes_3d,
                                         centers2d,
                                         depths,
                                         center_heatmap_pred.shape,
                                         img_metas[0]['pad_shape'],
                                         img_metas)

        center_heatmap_target = target_result['center_heatmap_target']
        offset_target = target_result['offset_target']
        dim_target = target_result['dim_target']
        depth_target = target_result['depth_target']
        alpha_cls_target = target_result['alpha_cls_target']
        alpha_offset_target = target_result['alpha_offset_target']

        indices = target_result['indices']

        mask_target = target_result['mask_target']

        # select desired preds and labels based on mask

        # 2d offset
        offset_pred = self.extract_input_from_tensor(offset_pred, indices, mask_target)
        offset_target = self.extract_target_from_tensor(offset_target, mask_target)
        # 3d dim
        dim_pred = self.extract_input_from_tensor(dim_pred, indices, mask_target)
        dim_target = self.extract_target_from_tensor(dim_target, mask_target)
        # depth
        depth_pred = self.extract_input_from_tensor(depth_pred, indices, mask_target)
        depth_target = self.extract_target_from_tensor(depth_target, mask_target)
        # alpha cls
        alpha_cls_pred = self.extract_input_from_tensor(alpha_cls_pred, indices, mask_target)
        alpha_cls_target = self.extract_target_from_tensor(alpha_cls_target, mask_target).type(torch.long)
        alpha_cls_onehot_target = alpha_cls_target.new_zeros([len(alpha_cls_target), self.num_alpha_bins]).scatter_(
            dim=1, index=alpha_cls_target.view(-1, 1), value=1)
        # alpha offset
        alpha_offset_pred = self.extract_input_from_tensor(alpha_offset_pred, indices, mask_target)
        alpha_offset_pred = torch.sum(alpha_offset_pred * alpha_cls_onehot_target, 1, keepdim=True)
        alpha_offset_target = self.extract_target_from_tensor(alpha_offset_target, mask_target)

        # calculate loss
        loss_center_heatmap = self.loss_center_heatmap(center_heatmap_pred, center_heatmap_target)

        loss_offset = self.loss_offset(offset_pred, offset_target)
        if self.dim_aware_in_loss:
            loss_dim = self.loss_dim(dim_pred, dim_target, dim_pred)
        else:
            loss_dim = self.loss_dim(dim_pred, dim_target)

        depth_pred, depth_log_variance = depth_pred[:, 0:1], depth_pred[:, 1:2]
        loss_depth = self.loss_depth(depth_pred, depth_target, depth_log_variance)

        if mask_target.sum() > 0:
            loss_alpha_cls = self.loss_alpha_cls(alpha_cls_pred, alpha_cls_onehot_target)
        else:
            loss_alpha_cls = 0.0
        loss_alpha_reg = self.loss_alpha_reg(alpha_offset_pred, alpha_offset_target)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_offset=loss_offset,
            loss_dim=loss_dim,
            loss_alpha_cls=loss_alpha_cls,
            loss_alpha_reg=loss_alpha_reg,
            loss_depth=loss_depth,
        )

    def get_targets(self, gt_bboxes, gt_labels,
                    gt_bboxes_3d,
                    centers2d,
                    depths,
                    feat_shape, img_shape,
                    img_metas):
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        # objects as 2D center points
        center_heatmap_target = gt_bboxes[-1].new_zeros([bs, self.num_classes, feat_h, feat_w])
        # 2D attributes
        offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 2])

        # 3D attributes
        dim_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 3])
        alpha_cls_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        alpha_offset_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        depth_target = gt_bboxes[-1].new_zeros([bs, self.max_objs, 1])
        # indices
        indices = gt_bboxes[-1].new_zeros([bs, self.max_objs]).type(torch.cuda.LongTensor)
        # masks
        mask_target = gt_bboxes[-1].new_zeros([bs, self.max_objs])
        for batch_id in range(bs):

            gt_bbox = gt_bboxes[batch_id]
            if len(gt_bbox) < 1:
                continue
            gt_label = gt_labels[batch_id]
            gt_bbox_3d = gt_bboxes_3d[batch_id]
            if not isinstance(gt_bbox_3d, torch.Tensor):
                gt_bbox_3d = gt_bbox_3d.tensor.to(gt_bbox.device)

            depth = depths[batch_id]
            gt_centers = centers2d[batch_id] * width_ratio
            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                
                is_in_feature = (0 <= ctx_int <= feat_w) and (0 <= cty_int <= feat_h)
                if not is_in_feature: continue

                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio

                dim = gt_bbox_3d[j][3: 6]
                alpha = gt_bbox_3d[j][6]

                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                indices[batch_id, j] = cty_int * feat_w + ctx_int

                offset_target[batch_id, j, 0] = ctx - ctx_int
                offset_target[batch_id, j, 1] = cty - cty_int

                dim_target[batch_id, j] = dim
                depth_target[batch_id, j] = depth[j]

                alpha_cls_target[batch_id, j], alpha_offset_target[batch_id, j] = self.angle2class(alpha)

                mask_target[batch_id, j] = 1

        mask_target = mask_target.type(torch.bool)

        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            offset_target=offset_target,
            dim_target=dim_target,
            depth_target=depth_target,
            alpha_cls_target=alpha_cls_target,
            alpha_offset_target=alpha_offset_target,
            indices=indices,
            mask_target=mask_target,
        )

        return target_result

    @staticmethod
    def extract_input_from_tensor(input, ind, mask):
        input = transpose_and_gather_feat(input, ind)
        return input[mask]

    @staticmethod
    def extract_target_from_tensor(target, mask):
        return target[mask]

    def angle2class(self, angle):
        ''' Convert continuous angle to discrete class and residual. '''
        angle = angle % (2 * PI)
        assert (angle >= 0 and angle <= 2 * PI)
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        shifted_angle = (angle + angle_per_class / 2) % (2 * PI)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, cls, residual):
        ''' Inverse function to angle2class. '''
        angle_per_class = 2 * PI / float(self.num_alpha_bins)
        angle_center = cls * angle_per_class
        angle = angle_center + residual
        return angle

    def decode_alpha_multibin(self, alpha_cls, alpha_offset):
        alpha_score, cls = alpha_cls.max(dim=-1)
        cls = cls.unsqueeze(2)
        alpha_offset = alpha_offset.gather(2, cls)
        alpha = self.class2angle(cls, alpha_offset)

        alpha[alpha > PI] = alpha[alpha > PI] - 2 * PI
        alpha[alpha < -PI] = alpha[alpha < -PI] + 2 * PI
        return alpha

    def get_bboxes(self,
                   center_heatmap_preds,
                   offset_preds,
                   dim_preds,
                   alpha_cls_preds,
                   alpha_offset_preds,
                   depth_preds,
                   img_metas,
                   rescale=False):

        assert len(center_heatmap_preds) == len(offset_preds) \
             == len(dim_preds) == len(alpha_cls_preds) == len(alpha_offset_preds) == 1
        box_type_3d = img_metas[0]['box_type_3d']
        trans_mat = torch.stack([
            offset_preds[0].new_tensor(img_meta['trans_mat'])
            for img_meta in img_metas
        ])

        batch_det_bboxes_3d, batch_labels = self.decode_heatmap(
            center_heatmap_preds[0],
            offset_preds[0],
            dim_preds[0],
            alpha_cls_preds[0],
            alpha_offset_preds[0],
            depth_preds[0],
            img_metas[0]['pad_shape'][:2],
            img_metas[0]['cam2img'],
            trans_mat,
            k=100,
            kernel=3,
            thresh=0.4)
        
        det_results = [
            [box_type_3d(batch_det_bboxes_3d[...,:-1],
                         box_dim=self.bbox_code_size, origin=(0.5, 0.5, 0.5)),
             batch_det_bboxes_3d[..., -1],
             batch_labels,
             ]
        ]
        return det_results

    def decode_heatmap(self,
                       center_heatmap_pred,
                       offset_pred,
                       dim_pred,
                       alpha_cls_pred,
                       alpha_offset_pred,
                       depth_pred,
                       img_shape,
                       camera_intrinsic,
                       trans_mat,
                       k=100,
                       kernel=3,
                       thresh=0.4):
        batch, cat, height, width = center_heatmap_pred.shape
        assert batch == 1
        inp_h, inp_w = img_shape

        _, _, feat_h, feat_w = center_heatmap_pred.shape
        down_ratio = inp_h / feat_h
        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)
        
        # (B, K)
        scores, indices, topk_labels, ys, xs = \
            get_topk_from_heatmap(center_heatmap_pred, k=k)
         
        
        offset = transpose_and_gather_feat(offset_pred, indices)       # (B, K, 2)
        
        # Get 3D Predictions from Predicted Heatmap
        # 'sigma' represents uncertainty.
        
        # Convert bin class and offset to alpha.
        alpha_cls = transpose_and_gather_feat(alpha_cls_pred, indices)         # (B, K, 12)
        alpha_offset = transpose_and_gather_feat(alpha_offset_pred, indices)   # (B, K, 12)
        alpha = self.decode_alpha_multibin(alpha_cls, alpha_offset)  # (b, k, 1)
        
        depth_pred = transpose_and_gather_feat(depth_pred, indices)            # (B, K, 2)


        sigma = torch.exp(-depth_pred[:, :, 1])                                             # (B, K)
        scores = scores[..., None]
        scores[..., -1] = (scores[..., -1] * sigma)
        
        points = torch.cat([xs.view(-1, 1),
                            ys.view(-1, 1).float()],
                           dim=1)


        points = points + offset.view(batch * k, -1) #仿射变换后图像的中心点位置
        points = self.inv_affine(points, trans_mat) #逆变换得到仿射变换前图像的中心点位置
        center2d = points.reshape(batch, k, -1)

        rot_y = self.recover_rotation(center2d, alpha, camera_intrinsic)  # (b, k, 3)
   
        depth = depth_pred[:, :, 0:1].view(batch, k, 1)                                                      # (B, K, 1)
        center3d = torch.cat([center2d, depth], dim=-1).squeeze(0)                                      # (B, K, 3)
        center3d = self.pts2Dto3D(center3d, np.array(camera_intrinsic)).unsqueeze(0)

        dim = transpose_and_gather_feat(dim_pred, indices).view(batch, k, 3)
        bboxes_3d = torch.cat([center3d, dim, rot_y, scores], dim=-1)
        
        box_mask = (scores[..., -1] > thresh)                                   # (B, K)

        bboxes_3d = bboxes_3d[box_mask]
        topk_labels = topk_labels[box_mask]
        return bboxes_3d, topk_labels

    def recover_rotation(self, kpts, alpha, calib):
        device = kpts.device
        calib = torch.tensor(calib).type(torch.FloatTensor).to(device).unsqueeze(0)

        si = torch.zeros_like(kpts[:, :, 0:1]) + calib[:, 0:1, 0:1]
        rot_y = alpha + torch.atan2(kpts[:, :, 0:1] - calib[:, 0:1, 2:3], si)

        while (rot_y > PI).any():
            rot_y[rot_y > PI] = rot_y[rot_y > PI] - 2 * PI
        while (rot_y < -PI).any():
            rot_y[rot_y < -PI] = rot_y[rot_y < -PI] + 2 * PI

        return rot_y

    def inv_affine(self, points, trans_mats):
        # number of points
        N = points.shape[0]
        # batch_size
        N_batch = trans_mats.shape[0]
        batch_id = torch.arange(N_batch).unsqueeze(1)
        obj_id = batch_id.repeat(1, N // N_batch).flatten()
        trans_mats_inv = trans_mats.inverse()[obj_id]
        centers2d_extend = torch.cat((points, points.new_ones(N, 1)),
                                     dim=1)
        # expand project points as [N, 3, 1]
        centers2d_extend = centers2d_extend.unsqueeze(-1)
        # transform project points back on original image
        centers2d_img = torch.matmul(trans_mats_inv, centers2d_extend).squeeze(2)

        return centers2d_img[:, :2]

    @staticmethod
    def pts2Dto3D(points, view):
        """
        Args:
            points (torch.Tensor): points in 2D images, [N, 3], \
                3 corresponds with x, y in the image and depth.
            view (np.ndarray): camera instrinsic, [3, 3]

        Returns:
            torch.Tensor: points in 3D space. [N, 3], \
                3 corresponds with x, y, z in 3D space.
        """
        assert view.shape[0] <= 4
        assert view.shape[1] <= 4
        assert points.shape[1] == 3

        points2D = points[:, :2]
        depths = points[:, 2].view(-1, 1)
        unnorm_points2D = torch.cat([points2D * depths, depths], dim=1)

        viewpad = torch.eye(4, dtype=points2D.dtype, device=points2D.device)
        viewpad[:view.shape[0], :view.shape[1]] = points2D.new_tensor(view)
        inv_viewpad = torch.inverse(viewpad).transpose(0, 1)

        # Do operation in homogenous coordinates.
        nbr_points = unnorm_points2D.shape[0]
        homo_points2D = torch.cat(
            [unnorm_points2D,
             points2D.new_ones((nbr_points, 1))], dim=1)
        points3D = torch.mm(homo_points2D, inv_viewpad)[:, :3]

        return points3D

    @staticmethod
    def _topk_channel(scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      centers2d=None,
                      depths=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        outs = self(x)
        assert gt_labels is not None
        assert attr_labels is None
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_bboxes_3d,
                              gt_labels_3d, centers2d, depths,
                              img_metas, attr_labels)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)

        if proposal_cfg is None:
            return losses
        else:
            raise NotImplementedError
