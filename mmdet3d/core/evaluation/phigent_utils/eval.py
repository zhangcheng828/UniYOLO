import numpy as np
def bev_box_overlap(boxes, qboxes, criterion=-1):
    from ..rotate_iou import rotate_iou_gpu_eval
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


def clean_data(gt_anno, dt_anno, difficulty, current_class='vehicle'):
    num_gt = len(gt_anno['name'])
    num_dt = len(dt_anno['name'])
    MIN_Depth = [15, 30, 60, 100]

    ignored_gt, ignored_dt = [], []
    
    num_valid_gt = 0
    num_valid_pred = 0
    for i in range(num_gt):
        
        depth = gt_anno['location'][i, -1]
        gt_name = gt_anno['name'][i].lower()

        valid_class = -1
        if (gt_name == current_class):
            valid_class = 1

        ignore = False
        if depth > MIN_Depth[difficulty] or (difficulty > 0 and depth <= MIN_Depth[difficulty - 1]):
            ignore = True

        if valid_class == 1 and ignore == False:
            ignored_gt.append(0)
            num_valid_gt += 1
        else:
            ignored_gt.append(1)

    for i in range(num_dt):
        if (dt_anno['name'][i].lower() == current_class):
            valid_class = 1
        else:
            valid_class = -1

        depth = dt_anno['location'][i, -1]
        ignore = False
        if (depth > MIN_Depth[difficulty] or (difficulty > 0 and depth <= MIN_Depth[difficulty - 1])):
            ignore = True
        
        if valid_class == 1 and ignore == False:
            ignored_dt.append(0)
            num_valid_pred += 1
        else:
            ignored_dt.append(1)

    return num_valid_gt, num_valid_pred, ignored_gt, ignored_dt



def eval_img(gt_anno, dt_anno, ignored_gt, ignored_det, pr,
             x_error_list, y_error_list, z_error_list, yaw_error_list, h_error_list, 
             w_error_list, l_error_list, rel_h_error_list, rel_w_error_list, 
             rel_l_error_list, iou_error_list):
    min_overlap = 0.5
    NO_DETECTION = -10000000
    
    loc = gt_anno['location'][:, [0, 2]]
    dims = gt_anno['dimensions'][:, [0, 2]]
    rots = gt_anno['rotation_y']
    gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                axis=1)
    loc = dt_anno['location'][:, [0, 2]]
    dims = dt_anno['dimensions'][:, [0, 2]]
    rots = dt_anno['rotation_y']
    dt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                axis=1)
    overlaps = bev_box_overlap(dt_boxes,
                                    gt_boxes).astype(np.float64)
    gt_size = gt_boxes.shape[0]
    det_size = dt_boxes.shape[0]
    assigned_detection = [False] * det_size

    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_det = False

        for j in range(det_size):
            if (ignored_det[j] == -1):
                continue
            if (assigned_detection[j]):
                continue
            overlap = overlaps[j, i]

            if ((overlap > min_overlap)
                  and (overlap > max_overlap or assigned_ignored_det)
                  and ignored_det[j] == 0):
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = False
            elif ((overlap > min_overlap)
                  and (valid_detection == NO_DETECTION)
                  and ignored_det[j] == 1):
                det_idx = j
                valid_detection = 1
                assigned_ignored_det = True

        if (valid_detection == NO_DETECTION) and ignored_gt[i] == 0:
            pr['fn'] += 1
        elif ((valid_detection != NO_DETECTION)
              and (ignored_gt[i] == 1 or ignored_det[det_idx] == 1)):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            pr['tp'] += 1
            assigned_detection[det_idx] = True
            x_error = abs(dt_anno['location'][det_idx, 0] - gt_anno['location'][i, 0])
            y_error = abs(dt_anno['location'][det_idx, 1] - gt_anno['location'][i, 1])
            z_error = abs(dt_anno['location'][det_idx, 2] - gt_anno['location'][i, 2])
            yaw_error = abs(dt_boxes[det_idx, -1] - gt_boxes[i, -1])
            h_error = abs(dt_anno['dimensions'][det_idx, 2] - gt_anno['dimensions'][i, 2])
            w_error = abs(dt_anno['dimensions'][det_idx, 1] - gt_anno['dimensions'][i, 1])
            l_error = abs(dt_anno['dimensions'][det_idx, 0] - gt_anno['dimensions'][i, 0])
            iou_error = 1 - overlaps[det_idx, i]

            rel_h_error = h_error / (gt_anno['dimensions'][i, 2] + 1e-5) * 100
            rel_w_error = w_error / (gt_anno['dimensions'][i, 1] + 1e-5) * 100
            rel_l_error = l_error / (gt_anno['dimensions'][i, 0] + 1e-5) * 100

            x_error_list.append(x_error)
            y_error_list.append(y_error)
            z_error_list.append(z_error)
            yaw_error_list.append(yaw_error)
            h_error_list.append(h_error)
            w_error_list.append(w_error)
            l_error_list.append(l_error)
            iou_error_list.append(iou_error)
            rel_h_error_list.append(rel_h_error)
            rel_w_error_list.append(rel_w_error)
            rel_l_error_list.append(rel_l_error)

    for i in range(det_size):
        if (not (assigned_detection[i] or ignored_det[i] == -1
                    or ignored_det[i] == 1)):
            pr['fp'] += 1


def do_eval(gt_annos,
            dt_annos,
            eval_types=['bev'],
            difficultys = [0, 1, 2, 3]):
    depth_difficultys = [[0 , 15], [15 , 30], [30, 60], [60, 100]]
    for difficulty in difficultys:
        pr = {'tp': 0, 'fp': 0, 'fn': 0}
        total_valid_gt, total_valid_pred = 0, 0
        x_error_list, y_error_list, z_error_list, yaw_error_list, h_error_list = [],[],[],[],[]
        w_error_list, l_error_list, rel_h_error_list, rel_w_error_list = [],[],[],[]
        rel_l_error_list, iou_error_list = [], []

        for i in range(len(gt_annos)):
            rets = clean_data(gt_annos[i], dt_annos[i], difficulty)
            num_valid_gt, num_valid_pred, ignored_gt, ignored_det = rets
            total_valid_gt += num_valid_gt
            total_valid_pred += num_valid_pred

            eval_img(gt_annos[i], dt_annos[i], ignored_gt, ignored_det, pr,
                    x_error_list, y_error_list, z_error_list, yaw_error_list, h_error_list, 
                    w_error_list, l_error_list, rel_h_error_list, rel_w_error_list, 
                    rel_l_error_list, iou_error_list)
        
        yaw_error_list = list(map(lambda x:(x/np.pi*180)%180, yaw_error_list))

        print('\ndepth@{}--{}m:'.format(depth_difficultys[difficulty][0], depth_difficultys[difficulty][1]))
        recall = pr['tp'] / (pr['tp'] + pr['fn'])
        precision = pr['tp']  / (pr['tp']  + pr['fp'] )

        print('metric   precision   recall   pred_cnt   gt_cnt')

        print('vehicle   {:.4f}     {:.4f}     {}     {}\n'.format(precision, recall, total_valid_pred, total_valid_gt))
        print('metric           avg     P50     P90     P95     P99')

        total_size = len(x_error_list)
        idx_50 = int(total_size * 0.5)
        idx_90 = int(total_size * 0.9)
        idx_95 = int(total_size * 0.95)
        idx_99 = int(total_size * 0.99)

        x_error_list.sort()
        y_error_list.sort()
        z_error_list.sort()
        yaw_error_list.sort()
        w_error_list.sort()
        h_error_list.sort()
        l_error_list.sort()
        rel_h_error_list.sort()
        rel_w_error_list.sort()
        rel_l_error_list.sort()
        iou_error_list.sort()


        print('      x_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(x_error_list)/total_size, 
                x_error_list[idx_50], x_error_list[idx_90], x_error_list[idx_95], x_error_list[idx_99]))
        print('      y_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(y_error_list)/total_size, 
                y_error_list[idx_50], y_error_list[idx_90], y_error_list[idx_95], y_error_list[idx_99]))
        print('      z_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(z_error_list)/total_size, 
                z_error_list[idx_50], z_error_list[idx_90], z_error_list[idx_95], z_error_list[idx_99]))
        print('    yaw_error(°): {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(yaw_error_list)/total_size, 
                yaw_error_list[idx_50], yaw_error_list[idx_90], yaw_error_list[idx_95], yaw_error_list[idx_99]))
        print('      h_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(h_error_list)/total_size, 
                h_error_list[idx_50], h_error_list[idx_90], h_error_list[idx_95], h_error_list[idx_99]))
        print('      w_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(w_error_list)/total_size, 
                w_error_list[idx_50], w_error_list[idx_90], w_error_list[idx_95], w_error_list[idx_99]))
        print('      l_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(l_error_list)/total_size, 
                l_error_list[idx_50], l_error_list[idx_90], l_error_list[idx_95], l_error_list[idx_99]))
        print('rel_h_error(%):{:.5}  {:.5}  {:.5}  {:.5}  {:.5}'.format(sum(rel_h_error_list)/total_size, 
                rel_h_error_list[idx_50], rel_h_error_list[idx_90], rel_h_error_list[idx_95], rel_h_error_list[idx_99]))
        print('rel_w_error(%):{:.5}  {:.5}  {:.5}  {:.5}  {:.5}'.format(sum(rel_w_error_list)/total_size, 
                rel_w_error_list[idx_50], rel_w_error_list[idx_90], rel_w_error_list[idx_95], rel_w_error_list[idx_99]))
        print('rel_l_error(%):{:.5}  {:.5}  {:.5}  {:.5}  {:.5}'.format(sum(rel_l_error_list)/total_size, 
                rel_l_error_list[idx_50], rel_l_error_list[idx_90], rel_l_error_list[idx_95], rel_l_error_list[idx_99]))
        print('    iou_error: {:.4f}  {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(sum(iou_error_list)/total_size, 
                iou_error_list[idx_50], iou_error_list[idx_90], iou_error_list[idx_95], iou_error_list[idx_99]))
        print('-----------------------------------------------')




def phigent_eval(gt_annos,
                 dt_annos,
                 current_classes,
                 eval_types=['bbox', 'bev', '3d']):
    """KITTI evaluation.
    Args:
        gt_annos (list[dict]): Contain gt information of each sample.
        dt_annos (list[dict]): Contain detected information of each sample.
        current_classes (list[str]): Classes to evaluation.
        eval_types (list[str], optional): Types to eval.
            Defaults to ['bbox', 'bev', '3d'].
    Returns:
        tuple: String and dict of evaluation results.
    """
    assert len(eval_types) > 0, 'must contain at least one evaluation type'

    do_eval(gt_annos, dt_annos)
    ret_dict = {}
    return ret_dict
