# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.nn import functional as F

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS

INF = 1e8
EPS = 1e-12
PI = np.pi

@BBOX_CODERS.register_module()
class UniYOLOCoder(BaseBBoxCoder):
    """Bbox Coder for MonoFlex.

    Args:
        depth_mode (str): The mode for depth calculation.
            Available options are "linear", "inv_sigmoid", and "exp".
        base_depth (tuple[float]): References for decoding box depth.
        depth_range (list): Depth range of predicted depth.
        combine_depth (bool): Whether to use combined depth (direct depth
            and depth from keypoints) or use direct depth only.
        uncertainty_range (list): Uncertainty range of predicted depth.
        base_dims (tuple[tuple[float]]): Dimensions mean and std of decode bbox
            dimensions [l, h, w] for each category.
        dims_mode (str): The mode for dimension calculation.
            Available options are "linear" and "exp".
        multibin (bool): Whether to use multibin representation.
        num_dir_bins (int): Number of Number of bins to encode
            direction angle.
        bin_centers (list[float]): Local yaw centers while using multibin
            representations.
        bin_margin (float): Margin of multibin representations.
        code_size (int): The dimension of boxes to be encoded.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-3.
    """

    def __init__(self,
                 depth_mode='linear',
                 num_alpha_bins=12,
                 eps=1e-3):
        super(UniYOLOCoder, self).__init__()
        # output related
        self.num_alpha_bins = num_alpha_bins
        self.depth_mode = depth_mode
        self.eps = eps

    def decode_bboxes2d(self, bbox_preds, priors):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes

    def decode_bboxes3d(self, bbox_3d_preds, alpha_preds, priors, cam2imgs):
        direct_depth = bbox_3d_preds[..., 2]
        direct_depth_log_var = bbox_3d_preds[..., 3]
        dims = bbox_3d_preds[..., 4:].view(-1, 3)
        depth = self.decode_depth(direct_depth)
        sigma = torch.exp(-direct_depth_log_var)

        center2d = (bbox_3d_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        # 1. decode alpha
        alpha = self.decode_alpha_multibin(alpha_preds[..., :self.num_alpha_bins], 
                                            alpha_preds[..., self.num_alpha_bins:])  # (b, k, 1)

        # 2. recover rotY
        rot_y = self.recover_rotation(center2d, alpha, cam2imgs)  # (b, k, 1)

        # 2.5 recover box3d_center from center2d and depth
        center3d = torch.cat([center2d, depth.view(1, -1, 1)], dim=-1).squeeze(0)
        center3d = self.pts2Dto3D(center3d, np.array(cam2imgs)).unsqueeze(0)

        # 3. compose 3D box
        batch_bboxes_3d = torch.cat([center3d, bbox_3d_preds[..., 4:], rot_y], dim=-1)

        return batch_bboxes_3d, sigma.view(1, -1)

    def decode_depth(self, direct_depth):
        
        if self.depth_mode == 'exp':
            direct_depth = direct_depth.exp()
        elif self.depth_mode == 'scale':
            direct_depth = direct_depth * 64 / priors[:, 3:4]
        elif self.depth_mode == 'inv_sigmoid':
            direct_depth = 1 / torch.sigmoid(direct_depth) - 1

        return direct_depth

    def pts2Dto3D(self, points, view):
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

    def encode(self, gt_bboxes_3d):
        pass

    def decode(self, bbox, base_centers2d, labels, downsample_ratio, cam2imgs):
        pass