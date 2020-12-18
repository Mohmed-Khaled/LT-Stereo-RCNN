# --------------------------------------------------------
# Faster RCNN implemented by Jiasen Lu, Jianwei Yang
# --------------------------------------------------------
# Modified by Peiliang Li for Stereo RCNN
# --------------------------------------------------------
# Modified by Mohamed Khaled
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.roi_layers import ROIAlign
from model.rpn.proposal_target_layer import _ProposalTargetLayer
from model.rpn.stereo_rpn import _Stereo_RPN
from model.utils.config import cfg
from model.utils.net_utils import _smooth_l1_loss
from torch.autograd import Variable
import torchvision.models as models


class _LTStereoRCNN(nn.Module):
    """ FPN """
    def __init__(self, classes):
        super(_LTStereoRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)

        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_dim = 0
        self.RCNN_loss_dim_orien = 0
        self.RCNN_loss_kpts = 0

        #define base network
        self.RCNN_base = models.mobilenet_v2(pretrained=True).features
        self.RCNN_base_reduce = nn.Conv2d(1280, 256, kernel_size=1, stride=1, padding=0)
        for p in self.RCNN_base.parameters(): p.requires_grad=False
        self.dout_base_model = 256

        # define rpn
        self.RCNN_rpn = _Stereo_RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)

        self.RCNN_roi_align = ROIAlign((cfg.POOLING_SIZE, cfg.POOLING_SIZE), 1.0/16.0, 0)
        self.RCNN_roi_kpts_align = ROIAlign((cfg.POOLING_SIZE*2, cfg.POOLING_SIZE*2), 1.0/16.0, 0)
    
    def _init_modules(self):
        self.RCNN_top = nn.Sequential(
            nn.Conv2d(512, 2048, kernel_size=cfg.POOLING_SIZE, stride=cfg.POOLING_SIZE, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Conv2d(2048, 2048, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Dropout(p=0.2)
        )

        self.RCNN_kpts = nn.Sequential(
            nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.dout_base_model, self.dout_base_model, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dout_base_model, self.dout_base_model, kernel_size=2, stride=2),
            nn.ReLU(True)
        )

        self.RCNN_cls_score = nn.Linear(2048, self.n_classes)

        self.RCNN_bbox_pred = nn.Linear(2048, 6*self.n_classes)
        self.RCNN_dim_orien_pred = nn.Linear(2048, 5*self.n_classes)
        self.kpts_class = nn.Conv2d(self.dout_base_model, 6, kernel_size=1, stride=1, padding=0)

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        # custom weights initialization called on netG and netD
        def weights_init(m, mean, stddev, truncated=False):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred_left_right, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_dim_orien_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)
        normal_init(self.kpts_class, 0, 0.1, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_top, 0, 0.01, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_kpts, 0, 0.1, cfg.TRAIN.TRUNCATED)
        weights_init(self.RCNN_base_reduce, 0, 0.1, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()

    def RoI_Feat(self, feat_map, rois, im_info, kpts=False, single_level=None):
        roi_pool_feats = []
        scale = feat_map.size(2) / im_info[0][0]
        if kpts is True:
            feat = self.RCNN_roi_kpts_align(feat_map, rois, scale)
        else:
            feat = self.RCNN_roi_align(feat_map, rois, scale)
        roi_pool_feats.append(feat)
        roi_pool_feat = torch.cat(roi_pool_feats, 0)
            
        return roi_pool_feat
    
    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        if mode:
            # Set fixed blocks to be in eval mode
            self.RCNN_base.eval()

    def _head_to_tail(self, pool5):
        block5 = self.RCNN_top(pool5)
        fc7 = block5.mean(3).mean(2)
        return fc7

    def forward(self, im_left_data, im_right_data, im_info, gt_boxes_left, gt_boxes_right,
                gt_boxes_merge, gt_dim_orien, gt_kpts, num_boxes):
        batch_size = im_left_data.size(0)

        im_info = im_info.data
        gt_boxes_left = gt_boxes_left.data
        gt_boxes_right = gt_boxes_right.data
        gt_boxes_merge = gt_boxes_merge.data
        gt_dim_orien = gt_dim_orien.data
        gt_kpts = gt_kpts.data
        num_boxes = num_boxes.data

        ## feed left image data to base model to obtain base feature map
        feat_left = self.RCNN_base_reduce(self.RCNN_base(im_left_data))

        ## feed right image data to base model to obtain base feature map
        feat_right = self.RCNN_base_reduce(self.RCNN_base(im_right_data))

        rois_left, rois_right, rpn_loss_cls, rpn_loss_bbox_left_right = \
            self.RCNN_rpn(feat_left, feat_right,
                          im_info, gt_boxes_left, gt_boxes_right, gt_boxes_merge, num_boxes)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois_left, rois_right, gt_boxes_left, gt_boxes_right,
                                                 gt_dim_orien, gt_kpts, num_boxes)
            rois_left, rois_right, rois_label, rois_target_left, rois_target_right,\
            rois_target_dim_orien, kpts_label_all, kpts_weight_all, rois_inside_ws4, rois_outside_ws4 = roi_data

            rois_target_left_right = rois_target_left.new(rois_target_left.size()[0],rois_target_left.size()[1],6)
            rois_target_left_right[:,:,:4] = rois_target_left
            rois_target_left_right[:,:,4] = rois_target_right[:,:,0]
            rois_target_left_right[:,:,5] = rois_target_right[:,:,2]

            rois_inside_ws = rois_inside_ws4.new(rois_inside_ws4.size()[0],rois_inside_ws4.size()[1],6)
            rois_inside_ws[:,:,:4] = rois_inside_ws4
            rois_inside_ws[:,:,4:] = rois_inside_ws4[:,:,0:2]

            rois_outside_ws = rois_outside_ws4.new(rois_outside_ws4.size()[0],rois_outside_ws4.size()[1],6)
            rois_outside_ws[:,:,:4] = rois_outside_ws4
            rois_outside_ws[:,:,4:] = rois_outside_ws4[:,:,0:2]

            rois_label = rois_label.view(-1).long()
            rois_label = Variable(rois_label)
            kpts_label = Variable(kpts_label_all[:,:,0].contiguous().view(-1))
            left_border_label = Variable(kpts_label_all[:,:,1].contiguous().view(-1))
            right_border_label = Variable(kpts_label_all[:,:,2].contiguous().view(-1))

            kpts_weight = Variable(kpts_weight_all[:,:,0].contiguous().view(-1))
            left_border_weight = Variable(kpts_weight_all[:,:,1].contiguous().view(-1))
            right_border_weight = Variable(kpts_weight_all[:,:,2].contiguous().view(-1))

            rois_target_left_right = Variable(rois_target_left_right.view(-1, rois_target_left_right.size(2)))
            rois_target_dim_orien = Variable(rois_target_dim_orien.view(-1, rois_target_dim_orien.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target_left_right = None
            rois_target_dim_orien = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois_left = rois_left.view(-1,5)
        rois_right = rois_right.view(-1,5)
        rois_left = Variable(rois_left)
        rois_right = Variable(rois_right)

        # pooling features based on rois, output 14x14 map
        roi_feat_semantic = torch.cat((self.RoI_Feat(feat_left, rois_left, im_info),
                                       self.RoI_Feat(feat_right, rois_right, im_info)),1)

        # feed pooled features to top model
        roi_feat_semantic = self._head_to_tail(roi_feat_semantic)
        bbox_pred = self.RCNN_bbox_pred(roi_feat_semantic)            # num x 6
        dim_orien_pred = self.RCNN_dim_orien_pred(roi_feat_semantic)  # num x 5

        cls_score = self.RCNN_cls_score(roi_feat_semantic)
        cls_prob = F.softmax(cls_score, 1) 

        # for keypoint
        roi_feat_dense = self.RoI_Feat(feat_left, rois_left, im_info, kpts=True)
        roi_feat_dense = self.RCNN_kpts(roi_feat_dense) # num x 256 x 28 x 28
        kpts_pred_all = self.kpts_class(roi_feat_dense) # num x 6 x cfg.KPTS_GRID x cfg.KPTS_GRID
        kpts_pred_all = kpts_pred_all.sum(2)            # num x 6 x cfg.KPTS_GRID
        kpts_pred = kpts_pred_all[:,:4,:].contiguous().view(-1, 4*cfg.KPTS_GRID)
        kpts_prob = F.softmax(kpts_pred,1) # num x (4xcfg.KPTS_GRID) 

        left_border_pred = kpts_pred_all[:,4,:].contiguous().view(-1, cfg.KPTS_GRID)
        left_border_prob = F.softmax(left_border_pred,1) # num x cfg.KPTS_GRID

        right_border_pred = kpts_pred_all[:,5,:].contiguous().view(-1, cfg.KPTS_GRID)
        right_border_prob = F.softmax(right_border_pred,1) # num x cfg.KPTS_GRID

        if self.training:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1)/6), 6) # (128L, 2L, 6L)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 6))
            bbox_pred = bbox_pred_select.squeeze(1)

            dim_orien_pred_view = dim_orien_pred.view(dim_orien_pred.size(0), int(dim_orien_pred.size(1)/5), 5) # (128L, 4L, 5L)
            dim_orien_pred_select = torch.gather(dim_orien_pred_view, 1, rois_label.long().view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 5))
            dim_orien_pred = dim_orien_pred_select.squeeze(1) 

        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0 

        if self.training:
            # classification loss
            self.RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            self.RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target_left_right, rois_inside_ws, rois_outside_ws)
            self.RCNN_loss_dim_orien = _smooth_l1_loss(dim_orien_pred, rois_target_dim_orien)
            
            kpts_pred = kpts_pred.view(-1, 4*cfg.KPTS_GRID)
            kpts_label = kpts_label.view(-1)
            self.RCNN_loss_kpts = F.cross_entropy(kpts_pred, kpts_label, reduce=False)
            if torch.sum(kpts_weight).item() < 1:
                self.RCNN_loss_kpts = torch.sum(self.RCNN_loss_kpts*kpts_weight)
            else:
                self.RCNN_loss_kpts = torch.sum(self.RCNN_loss_kpts*kpts_weight)/torch.sum(kpts_weight)

            self.RCNN_loss_left_border = F.cross_entropy(left_border_pred, left_border_label, reduce=False)
            if torch.sum(left_border_weight).item() < 1:
                self.RCNN_loss_left_border = torch.sum(self.RCNN_loss_left_border*left_border_weight)
            else:
                self.RCNN_loss_left_border = torch.sum(self.RCNN_loss_left_border*left_border_weight)/torch.sum(left_border_weight)

            self.RCNN_loss_right_border = F.cross_entropy(right_border_pred, right_border_label, reduce=False)
            if torch.sum(right_border_weight).item()<1:
                self.RCNN_loss_right_border = torch.sum(self.RCNN_loss_right_border*right_border_weight)
            else:
                self.RCNN_loss_right_border = torch.sum(self.RCNN_loss_right_border*right_border_weight)/torch.sum(right_border_weight)
            self.RCNN_loss_kpts = (self.RCNN_loss_kpts+self.RCNN_loss_left_border+self.RCNN_loss_right_border)/3.0

        rois_left = rois_left.view(batch_size,-1, rois_left.size(1))
        rois_right = rois_right.view(batch_size, -1, rois_right.size(1))
        cls_prob = cls_prob.view(batch_size, -1, cls_prob.size(1))
        bbox_pred = bbox_pred.view(batch_size, -1, bbox_pred.size(1))
        dim_orien_pred = dim_orien_pred.view(batch_size, -1, dim_orien_pred.size(1)) 

        if self.training:
            rois_label = rois_label.view(batch_size, -1)

        return rois_left, rois_right, cls_prob, bbox_pred, dim_orien_pred, \
               kpts_prob, left_border_prob, right_border_prob, rpn_loss_cls, rpn_loss_bbox_left_right, \
               self.RCNN_loss_cls, self.RCNN_loss_bbox, self.RCNN_loss_dim_orien, self.RCNN_loss_kpts, rois_label  









