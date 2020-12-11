import cv2
import numpy as np
import torch
from model.rpn.bbox_transform import bbox_transform_inv, kpts_transform_inv, border_transform_inv, clip_boxes
from model.utils.config import cfg
from model.rt_stereo_rcnn.rt_stereo_rcnn import _RTStereoRCNN
import time

if __name__ == "__main__":
    np.random.seed(cfg.RNG_SEED)

    kitti_classes = np.asarray(['__background__', 'Car'])

    # initilize the network here.
    stereoRCNN = _RTStereoRCNN(kitti_classes)
    stereoRCNN.create_architecture()

    with torch.no_grad():
        stereoRCNN.eval()

        num_params = sum(p.numel() for p in stereoRCNN.parameters())
        num_params_train = sum(p.numel() for p in stereoRCNN.parameters() if p.requires_grad)
        print("Number of parameters: {}".format(num_params))
        print("Number of trainable parameters: {}".format(num_params_train))
        print("Number of freezed parameters: {}".format(num_params - num_params_train))

        im_left_data = torch.FloatTensor(1)
        im_right_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # read data
        img_l_path = 'demo/left.png'
        img_r_path = 'demo/right.png'

        img_left = cv2.imread(img_l_path)
        img_right = cv2.imread(img_r_path)

        # rgb -> bgr
        img_left = img_left.astype(np.float32, copy=False)
        img_right = img_right.astype(np.float32, copy=False)

        img_left -= cfg.PIXEL_MEANS
        img_right -= cfg.PIXEL_MEANS

        im_shape = img_left.shape
        im_size_min = np.min(im_shape[0:2])
        im_scale = float(cfg.TRAIN.SCALES[0]) / float(im_size_min)

        img_left = cv2.resize(img_left, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right, None, None, fx=im_scale, fy=im_scale,
                                interpolation=cv2.INTER_LINEAR)

        info = np.array([[img_left.shape[0], img_left.shape[1],
                            im_scale]], dtype=np.float32)

        img_left = torch.from_numpy(img_left)
        img_left = img_left.permute(2, 0, 1).unsqueeze(0).contiguous()

        img_right = torch.from_numpy(img_right)
        img_right = img_right.permute(2, 0, 1).unsqueeze(0).contiguous()

        info = torch.from_numpy(info)

        im_left_data.resize_(img_left.size()).copy_(img_left)
        im_right_data.resize_(img_right.size()).copy_(img_right)
        im_info.resize_(info.size()).copy_(info)

        det_tic = time.time()
        rois_left, rois_right, cls_prob, bbox_pred, bbox_pred_dim, kpts_prob, \
        left_prob, right_prob, rpn_loss_cls, rpn_loss_box_left_right, \
        RCNN_loss_cls, RCNN_loss_bbox, RCNN_loss_dim_orien, RCNN_loss_kpts, rois_label = \
            stereoRCNN(im_left_data, im_right_data, im_info, gt_boxes, gt_boxes,
                    gt_boxes, gt_boxes, gt_boxes, num_boxes)

        scores = cls_prob.data
        boxes_left = rois_left.data[:, :, 1:5]
        boxes_right = rois_right.data[:, :, 1:5]

        bbox_pred = bbox_pred.data
        box_delta_left = bbox_pred.new(bbox_pred.size()[1], 4 * len(kitti_classes)).zero_()
        box_delta_right = bbox_pred.new(bbox_pred.size()[1], 4 * len(kitti_classes)).zero_()

        for keep_inx in range(box_delta_left.size()[0]):
            box_delta_left[keep_inx, 0::4] = bbox_pred[0, keep_inx, 0::6]
            box_delta_left[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
            box_delta_left[keep_inx, 2::4] = bbox_pred[0, keep_inx, 2::6]
            box_delta_left[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

            box_delta_right[keep_inx, 0::4] = bbox_pred[0, keep_inx, 4::6]
            box_delta_right[keep_inx, 1::4] = bbox_pred[0, keep_inx, 1::6]
            box_delta_right[keep_inx, 2::4] = bbox_pred[0, keep_inx, 5::6]
            box_delta_right[keep_inx, 3::4] = bbox_pred[0, keep_inx, 3::6]

        box_delta_left = box_delta_left.view(-1, 4)
        box_delta_right = box_delta_right.view(-1, 4)

        dim_orien = bbox_pred_dim.data
        dim_orien = dim_orien.view(-1, 5)

        kpts_prob = kpts_prob.data
        kpts_prob = kpts_prob.view(-1, 4 * cfg.KPTS_GRID)
        max_prob, kpts_delta = torch.max(kpts_prob, 1)

        left_prob = left_prob.data
        left_prob = left_prob.view(-1, cfg.KPTS_GRID)
        _, left_delta = torch.max(left_prob, 1)

        right_prob = right_prob.data
        right_prob = right_prob.view(-1, cfg.KPTS_GRID)
        _, right_delta = torch.max(right_prob, 1)

        box_delta_left = box_delta_left * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        box_delta_right = box_delta_right * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                        + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
        dim_orien = dim_orien * torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_STDS) \
                    + torch.FloatTensor(cfg.TRAIN.DIM_NORMALIZE_MEANS)

        box_delta_left = box_delta_left.view(1, -1, 4 * len(kitti_classes))
        box_delta_right = box_delta_right.view(1, -1, 4 * len(kitti_classes))
        dim_orien = dim_orien.view(1, -1, 5 * len(kitti_classes))
        kpts_delta = kpts_delta.view(1, -1, 1)
        left_delta = left_delta.view(1, -1, 1)
        right_delta = right_delta.view(1, -1, 1)
        max_prob = max_prob.view(1, -1, 1)

        pred_boxes_left = bbox_transform_inv(boxes_left, box_delta_left, 1)
        pred_boxes_right = bbox_transform_inv(boxes_right, box_delta_right, 1)
        pred_kpts, kpts_type = kpts_transform_inv(boxes_left, kpts_delta, cfg.KPTS_GRID)
        pred_left = border_transform_inv(boxes_left, left_delta, cfg.KPTS_GRID)
        pred_right = border_transform_inv(boxes_left, right_delta, cfg.KPTS_GRID)

        pred_boxes_left = clip_boxes(pred_boxes_left, im_info.data, 1)
        pred_boxes_right = clip_boxes(pred_boxes_right, im_info.data, 1)

        pred_boxes_left /= im_info[0, 2].data
        pred_boxes_right /= im_info[0, 2].data
        pred_kpts /= im_info[0, 2].data
        pred_left /= im_info[0, 2].data
        pred_right /= im_info[0, 2].data

        scores = scores.squeeze()
        pred_boxes_left = pred_boxes_left.squeeze()
        pred_boxes_right = pred_boxes_right.squeeze()

        pred_kpts = torch.cat((pred_kpts, kpts_type, max_prob, pred_left, pred_right), 2)
        pred_kpts = pred_kpts.squeeze()
        dim_orien = dim_orien.squeeze()

        det_toc = time.time()
        detect_time = det_toc - det_tic

        print('Demo mode: det_time {:.2f}s'.format(detect_time))