import torch
import torch.nn as nn
import torch_scatter
import torch_cluster
from peft import LoraConfig, get_peft_model
from collections import OrderedDict

from pointcept.models.losses import build_criteria
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2batch
from .builder import MODELS, build_model


@MODELS.register_module()
class DefaultSegmentor(nn.Module):
    def __init__(self, backbone=None, criteria=None):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)

    def forward(self, input_dict):
        if "condition" in input_dict.keys():
            # PPT (https://arxiv.org/abs/2308.09718)
            # currently, only support one batch one condition
            input_dict["condition"] = input_dict["condition"][0]
        seg_logits = self.backbone(input_dict)
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss)
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return dict(loss=loss, seg_logits=seg_logits)
        # test
        else:
            return dict(seg_logits=seg_logits)


@MODELS.register_module()
class DefaultSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat and use DefaultSegmentorV2
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class CamGuideSegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        cam_loss_weight=0.0,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.cam_loss_weight = cam_loss_weight
        self.img_teacher = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()

        if return_point:
            return_dict["point"] = point
        if self.training:
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            loss_cam = seg_logits.new_zeros(())
            has_cam = (
                "images" in input_dict
                and "correspondence" in input_dict
                and input_dict["images"].numel() > 0
                and input_dict["correspondence"].shape[1] > 0
            )
            if has_cam:
                corr = input_dict["correspondence"]
                valid = (corr[..., 0] >= 0) & (corr[..., 1] >= 0)
                img = input_dict["images"]
                rgb_sum = seg_logits.new_zeros((corr.shape[0], 3))
                rgb_cnt = seg_logits.new_zeros((corr.shape[0], 1))
                for j in range(corr.shape[1]):
                    vj = valid[:, j]
                    if vj.any():
                        uv = corr[vj, j].long()
                        rgb = img[j, :, uv[:, 1], uv[:, 0]].permute(1, 0)
                        rgb_sum[vj] += rgb
                        rgb_cnt[vj] += 1.0
                keep = rgb_cnt[:, 0] > 0
                if keep.any():
                    rgb_mean = rgb_sum[keep] / rgb_cnt[keep]
                    teacher_logits = self.img_teacher(rgb_mean)
                    student_logits = seg_logits[keep]
                    loss_cam = torch.nn.functional.kl_div(
                        torch.log_softmax(student_logits, dim=-1),
                        torch.softmax(teacher_logits.detach(), dim=-1),
                        reduction="batchmean",
                    )
            return_dict["loss"] = loss_seg + self.cam_loss_weight * loss_cam
            return_dict["loss_seg"] = loss_seg
            return_dict["loss_cam"] = loss_cam
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits

        return return_dict
@MODELS.register_module()
class DefaultLORASegmentorV2(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        use_lora=False,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        backbone_path=None,
        keywords=None,
        replacements=None,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.keywords = keywords
        self.replacements = replacements
        self.backbone = build_model(backbone)
        backbone_weight = torch.load(
            backbone_path,
            map_location=lambda storage, loc: storage.cuda(),
        )
        self.backbone_load(backbone_weight)

        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        self.use_lora = use_lora

        if self.use_lora:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["qkv"],
                # target_modules=["query", "value"],
                lora_dropout=lora_dropout,
                bias="none",
            )
            self.backbone.enc = get_peft_model(self.backbone.enc, lora_config)

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        if self.use_lora:
            for name, param in self.backbone.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        self.backbone.enc.print_trainable_parameters()

    def backbone_load(self, checkpoint):
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            if self.keywords in key:
                key = key.replace(self.keywords, self.replacements)
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
            if key.startswith("backbone."):
                key = key[9:]
            weight[key] = value
        load_state_info = self.backbone.load_state_dict(weight, strict=False)
        print(f"Missing keys: {load_state_info[0]}")
        print(f"Unexpected keys: {load_state_info[1]}")

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.freeze_backbone and not self.use_lora:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            return_dict["point"] = point

        if self.training:
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DINOEnhancedSegmentor(nn.Module):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
    ):
        super().__init__()
        self.seg_head = (
            nn.Linear(backbone_out_channels, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.backbone = build_model(backbone) if backbone is not None else None
        self.criteria = build_criteria(criteria)
        self.freeze_backbone = freeze_backbone
        if self.backbone is not None and self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        if self.backbone is not None:
            if self.freeze_backbone:
                with torch.no_grad():
                    point = self.backbone(point)
            else:
                point = self.backbone(point)
            point_list = [point]
            while "unpooling_parent" in point_list[-1].keys():
                point_list.append(point_list[-1].pop("unpooling_parent"))
            for i in reversed(range(1, len(point_list))):
                point = point_list[i]
                parent = point_list[i - 1]
                assert "pooling_inverse" in point.keys()
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = point_list[0]
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pooling_inverse
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = [point.feat]
        else:
            feat = []
        dino_coord = input_dict["dino_coord"]
        dino_feat = input_dict["dino_feat"]
        dino_offset = input_dict["dino_offset"]
        idx = torch_cluster.knn(
            x=dino_coord,
            y=point.origin_coord,
            batch_x=offset2batch(dino_offset),
            batch_y=offset2batch(point.origin_offset),
            k=1,
        )[1]

        feat.append(dino_feat[idx])
        feat = torch.concatenate(feat, dim=-1)
        seg_logits = self.seg_head(feat)
        return_dict = dict()
        if return_point:
            # PCA evaluator parse feat and coord in point
            return_dict["point"] = point
        # train
        if self.training:
            loss = self.criteria(seg_logits, input_dict["origin_segment"])
            return_dict["loss"] = loss
        # eval
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["origin_segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        # test
        else:
            return_dict["seg_logits"] = seg_logits
        return return_dict


@MODELS.register_module()
class DefaultClassifier(nn.Module):
    def __init__(
        self,
        backbone=None,
        criteria=None,
        num_classes=40,
        backbone_embed_dim=256,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.criteria = build_criteria(criteria)
        self.num_classes = num_classes
        self.backbone_embed_dim = backbone_embed_dim
        self.freeze_backbone = freeze_backbone
        self.cls_head = nn.Sequential(
            nn.Linear(backbone_embed_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
        )
        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def forward(self, input_dict):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)
        # Backbone added after v1.5.0 return Point instead of feat
        # And after v1.5.0 feature aggregation for classification operated in classifier
        # TODO: remove this part after make all backbone return Point only.
        if isinstance(point, Point):
            point.feat = torch_scatter.segment_csr(
                src=point.feat,
                indptr=nn.functional.pad(point.offset, (1, 0)),
                reduce="mean",
            )
            feat = point.feat
        else:
            feat = point
        cls_logits = self.cls_head(feat)
        if self.training:
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss)
        elif "category" in input_dict.keys():
            loss = self.criteria(cls_logits, input_dict["category"])
            return dict(loss=loss, cls_logits=cls_logits)
        else:
            return dict(cls_logits=cls_logits)

@MODELS.register_module()
class MOCHAPretrainerV1(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        backbone=None,
        freeze_backbone=False,
        cam_loss_weight=1.0,
        proj_dim=64,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.freeze_backbone = freeze_backbone
        self.cam_loss_weight = cam_loss_weight

        self.student_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.img_teacher = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, proj_dim),
        )

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _extract_feat(self, input_dict):
        point = Point(input_dict)
        if self.freeze_backbone:
            with torch.no_grad():
                point = self.backbone(point)
        else:
            point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point
        return point, feat


    def forward(self, input_dict, return_point=False):
        point, feat = self._extract_feat(input_dict)
        return_dict = dict()

        if return_point:
            return_dict["point"] = point
        student_feat = self.student_proj(feat)
        student_feat = torch.nn.functional.normalize(student_feat, dim=-1)

        loss_cam = student_feat.sum() * 0.0
        matched_points = student_feat.new_zeros((), dtype=torch.long)

        has_cam = (
            "images" in input_dict
            and "correspondence" in input_dict
            and input_dict["images"].numel() > 0
            and input_dict["correspondence"].shape[1] > 0
        )
        if has_cam:
            corr = input_dict["correspondence"]
            valid = (corr[..., 0] >= 0) & (corr[..., 1] >= 0)
            img = input_dict["images"]

            rgb_sum = student_feat.new_zeros((corr.shape[0], 3))
            rgb_cnt = student_feat.new_zeros((corr.shape[0], 1))
            for j in range(corr.shape[1]):
                vj = valid[:, j]
                if vj.any():
                    uv = corr[vj, j].long()
                    rgb = img[j, :, uv[:, 1], uv[:, 0]].permute(1, 0)
                    rgb_sum[vj] += rgb
                    rgb_cnt[vj] += 1.0
            keep = rgb_cnt[:, 0] > 0
            matched_points = keep.sum()
            if keep.any():
                rgb_mean = rgb_sum[keep] / rgb_cnt[keep]
                teacher_feat = self.img_teacher(rgb_mean)
                teacher_feat = torch.nn.functional.normalize(teacher_feat.detach(), dim=-1)
                student_keep = student_feat[keep]
                loss_cam = 1.0 - torch.nn.functional.cosine_similarity(
                    student_keep, teacher_feat, dim=-1
                ).mean()

        loss_ssl = loss_cam
        loss = self.cam_loss_weight * loss_ssl

        return_dict["loss"] = loss
        return_dict["loss_ssl"] = loss_ssl
        return_dict["loss_cam"] = loss_cam
        return_dict["matched_points"] = matched_points
        return return_dict

@MODELS.register_module()
class MOCHAPretrainerV2EMA(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        coarse_out_channels,
        backbone=None,
        freeze_backbone=False,
        ema_momentum=0.996,
        lidar_loss_weight=1.0,
        cam_loss_weight=0.25,
        proj_dim=128,
        student_coord_jitter=0.01,
        teacher_coord_jitter=0.002,
        color_drop_prob=0.2,
        cam_drop_prob=0.0,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.teacher_backbone = build_model(backbone)

        self.freeze_backbone = freeze_backbone
        self.ema_momentum = ema_momentum
        self.lidar_loss_weight = lidar_loss_weight
        self.cam_loss_weight = cam_loss_weight
        self.student_coord_jitter = student_coord_jitter
        self.teacher_coord_jitter = teacher_coord_jitter
        self.color_drop_prob = color_drop_prob
        self.cam_drop_prob = cam_drop_prob

        self.student_proj = nn.Sequential(
            nn.Linear(coarse_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.teacher_proj = nn.Sequential(
            nn.Linear(coarse_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.cam_student_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.img_teacher = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, proj_dim),
        )

        self.teacher_initialized = False

    @torch.no_grad()
    def _sync_teacher(self):
        self.teacher_backbone.load_state_dict(self.backbone.state_dict(), strict=True)
        self.teacher_proj.load_state_dict(self.student_proj.state_dict(), strict=True)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_proj.parameters():
            p.requires_grad = False
        self.teacher_initialized = True


    @torch.no_grad()
    def _ema_update(self):
        m = self.ema_momentum
        for p_s, p_t in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
        for p_s, p_t in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

    def _clone_aug_input(self, input_dict, coord_jitter_std=0.0, color_drop_prob=0.0):
        out = dict(input_dict)

        if "coord" in input_dict:
            coord = input_dict["coord"].clone()
            if coord_jitter_std > 0:
                coord = coord + torch.randn_like(coord) * coord_jitter_std
            out["coord"] = coord

        if "color" in input_dict:
            color = input_dict["color"].clone()
            if color_drop_prob > 0:
                keep = (torch.rand(color.shape[0], 1, device=color.device) > color_drop_prob).to(color.dtype)
                color = color * keep
            out["color"] = color

        return out

    def _extract_feats(self, model, input_dict):
        point = Point(input_dict)
        point = model(point)

        if isinstance(point, Point):
            coarse_feat = point.feat
            point_full = point
            while "pooling_parent" in point_full.keys():
                assert "pooling_inverse" in point_full.keys()
                parent = point_full.pop("pooling_parent")
                inverse = point_full.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_full.feat[inverse]], dim=-1)
                point_full = parent
            final_feat = point_full.feat
        else:
            coarse_feat = point
            final_feat = point

        return coarse_feat, final_feat

    def forward(self, input_dict, return_point=False):
        if not self.teacher_initialized:
            self._sync_teacher()
        else:
            self._ema_update()

        student_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.student_coord_jitter,
            color_drop_prob=self.color_drop_prob,
        )
        teacher_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.teacher_coord_jitter,
            color_drop_prob=0.0,
        )
        coarse_student, final_student = self._extract_feats(self.backbone, student_input)
        with torch.no_grad():
            coarse_teacher, _ = self._extract_feats(self.teacher_backbone, teacher_input)

        student_lidar = torch.nn.functional.normalize(self.student_proj(coarse_student), dim=-1)
        with torch.no_grad():
            teacher_lidar = torch.nn.functional.normalize(self.teacher_proj(coarse_teacher), dim=-1)

        loss_lidar = 1.0 - torch.nn.functional.cosine_similarity(
            student_lidar, teacher_lidar, dim=-1
        ).mean()
        student_cam = torch.nn.functional.normalize(self.cam_student_proj(final_student), dim=-1)
        loss_cam = student_cam.sum() * 0.0
        matched_points = student_cam.new_zeros((), dtype=torch.long)

        use_cam = (
            "images" in input_dict
            and "correspondence" in input_dict
            and input_dict["images"].numel() > 0
            and input_dict["correspondence"].shape[1] > 0
        )
        if self.training and self.cam_drop_prob > 0:
            drop_cam = torch.rand((), device=student_cam.device) < self.cam_drop_prob
            if drop_cam.item():
                use_cam = False
        if use_cam:
            corr = input_dict["correspondence"]
            valid = (corr[..., 0] >= 0) & (corr[..., 1] >= 0)
            img = input_dict["images"]

            rgb_sum = student_cam.new_zeros((corr.shape[0], 3))
            rgb_cnt = student_cam.new_zeros((corr.shape[0], 1))

            for j in range(corr.shape[1]):
                vj = valid[:, j]
                if vj.any():
                    uv = corr[vj, j].long()
                    rgb = img[j, :, uv[:, 1], uv[:, 0]].permute(1, 0)
                    rgb_sum[vj] += rgb
                    rgb_cnt[vj] += 1.0
            keep = rgb_cnt[:, 0] > 0
            matched_points = keep.sum()

            if keep.any():
                rgb_mean = rgb_sum[keep] / rgb_cnt[keep]
                teacher_img = self.img_teacher(rgb_mean)
                teacher_img = torch.nn.functional.normalize(teacher_img.detach(), dim=-1)
                student_keep = student_cam[keep]
                loss_cam = 1.0 - torch.nn.functional.cosine_similarity(
                    student_keep, teacher_img, dim=-1
                ).mean()

        loss_ssl = self.lidar_loss_weight * loss_lidar + self.cam_loss_weight * loss_cam

        return dict(
            loss=loss_ssl,
            loss_ssl=loss_ssl,
            loss_lidar=loss_lidar,
            loss_cam=loss_cam,
            matched_points=matched_points,
        )

@MODELS.register_module()
class MOCHAPretrainerV3Masked(nn.Module):
    def __init__(
        self,
        backbone_out_channels,
        coarse_out_channels,
        backbone=None,
        freeze_backbone=False,
        ema_momentum=0.996,
        lidar_loss_weight=1.0,
        cam_loss_weight=0.20,
        proj_dim=128,
        global_coord_jitter=0.01,
        strong_coord_jitter=0.02,
        teacher_coord_jitter=0.002,
        global_color_drop=0.2,
        strong_color_drop=0.4,
        global_mask_ratio=0.30,
        strong_mask_ratio=0.50,
        cam_drop_prob=0.25,
    ):
        super().__init__()
        self.backbone = build_model(backbone)
        self.teacher_backbone = build_model(backbone)

        self.freeze_backbone = freeze_backbone
        self.ema_momentum = ema_momentum
        self.lidar_loss_weight = lidar_loss_weight
        self.cam_loss_weight = cam_loss_weight
        self.global_coord_jitter = global_coord_jitter
        self.strong_coord_jitter = strong_coord_jitter
        self.teacher_coord_jitter = teacher_coord_jitter
        self.global_color_drop = global_color_drop
        self.strong_color_drop = strong_color_drop
        self.global_mask_ratio = global_mask_ratio
        self.strong_mask_ratio = strong_mask_ratio
        self.cam_drop_prob = cam_drop_prob

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.student_proj = nn.Sequential(
            nn.Linear(coarse_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.teacher_proj = nn.Sequential(
            nn.Linear(coarse_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.cam_student_proj = nn.Sequential(
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim),
        )
        self.img_teacher = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, proj_dim),
        )

        self.teacher_initialized = False

    @torch.no_grad()
    def _sync_teacher(self):
        self.teacher_backbone.load_state_dict(self.backbone.state_dict(), strict=True)
        self.teacher_proj.load_state_dict(self.student_proj.state_dict(), strict=True)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False
        for p in self.teacher_proj.parameters():
            p.requires_grad = False
        self.teacher_initialized = True

    @torch.no_grad()
    def _ema_update(self):
        m = self.ema_momentum
        for p_s, p_t in zip(self.backbone.parameters(), self.teacher_backbone.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)
        for p_s, p_t in zip(self.student_proj.parameters(), self.teacher_proj.parameters()):
            p_t.data.mul_(m).add_(p_s.data, alpha=1.0 - m)

    def _clone_aug_input(self, input_dict, coord_jitter_std=0.0, color_drop_prob=0.0):
        out = dict(input_dict)

        if "coord" in input_dict:
            coord = input_dict["coord"].clone()
            if coord_jitter_std > 0:
                coord = coord + torch.randn_like(coord) * coord_jitter_std
            out["coord"] = coord

        if "color" in input_dict:
            color = input_dict["color"].clone()
            if color_drop_prob > 0:
                keep = (torch.rand(color.shape[0], 1, device=color.device) > color_drop_prob).to(color.dtype)
                color = color * keep
            out["color"] = color

        return out

    def _extract_feats(self, model, input_dict):
        point = Point(input_dict)
        point = model(point)

        if isinstance(point, Point):
            coarse_feat = point.feat
            point_full = point
            while "pooling_parent" in point_full.keys():
                assert "pooling_inverse" in point_full.keys()
                parent = point_full.pop("pooling_parent")
                inverse = point_full.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point_full.feat[inverse]], dim=-1)
                point_full = parent
            final_feat = point_full.feat
        else:
            coarse_feat = point
            final_feat = point

        return coarse_feat, final_feat

    def _mask_feat(self, feat, mask_ratio=0.0):
        if mask_ratio <= 0:
            return feat
        keep = (torch.rand(feat.shape[0], 1, device=feat.device) > mask_ratio).to(feat.dtype)
        return feat * keep

    def _norm_proj(self, proj, feat):
        return torch.nn.functional.normalize(proj(feat), dim=-1)

    def forward(self, input_dict, return_point=False):
        if not self.teacher_initialized:
            self._sync_teacher()
        else:
            self._ema_update()

        g1_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.global_coord_jitter,
            color_drop_prob=self.global_color_drop,
        )
        s1_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.strong_coord_jitter,
            color_drop_prob=self.strong_color_drop,
        )
        teacher_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.teacher_coord_jitter,
            color_drop_prob=0.0,
        )
        g1_coarse, g1_final = self._extract_feats(self.backbone, g1_input)
        s1_coarse, _ = self._extract_feats(self.backbone, s1_input)

        with torch.no_grad():
            teacher_coarse, _ = self._extract_feats(self.teacher_backbone, teacher_input)

        teacher_feat = self._norm_proj(self.teacher_proj, teacher_coarse)

        views = [
            self._norm_proj(self.student_proj, g1_coarse),
            self._norm_proj(self.student_proj, s1_coarse),
            self._norm_proj(self.student_proj, self._mask_feat(g1_coarse, self.global_mask_ratio)),
            self._norm_proj(self.student_proj, self._mask_feat(s1_coarse, self.strong_mask_ratio)),
        ]

        loss_terms = []
        for v in views:
            loss_terms.append(
                1.0 - torch.nn.functional.cosine_similarity(v, teacher_feat, dim=-1).mean()
            )
        loss_lidar = torch.stack(loss_terms).mean()
        student_cam = self._norm_proj(self.cam_student_proj, g1_final)
        loss_cam = student_cam.sum() * 0.0
        matched_points = student_cam.new_zeros((), dtype=torch.long)

        use_cam = (
            self.cam_loss_weight > 0
            and "images" in input_dict
            and "correspondence" in input_dict
            and input_dict["images"].numel() > 0
            and input_dict["correspondence"].shape[1] > 0
        )
        if self.training and self.cam_drop_prob > 0:
            drop_cam = torch.rand((), device=student_cam.device) < self.cam_drop_prob
            if drop_cam.item():
                use_cam = False

        if use_cam:
            corr = input_dict["correspondence"]
            valid = (corr[..., 0] >= 0) & (corr[..., 1] >= 0)
            img = input_dict["images"]

            rgb_sum = student_cam.new_zeros((corr.shape[0], 3))
            rgb_cnt = student_cam.new_zeros((corr.shape[0], 1))

            for j in range(corr.shape[1]):
                vj = valid[:, j]
                if vj.any():
                    uv = corr[vj, j].long()
                    rgb = img[j, :, uv[:, 1], uv[:, 0]].permute(1, 0)
                    rgb_sum[vj] += rgb
                    rgb_cnt[vj] += 1.0

            keep = rgb_cnt[:, 0] > 0
            matched_points = keep.sum()

            if keep.any():
                rgb_mean = rgb_sum[keep] / rgb_cnt[keep]
                teacher_img = self.img_teacher(rgb_mean)
                teacher_img = torch.nn.functional.normalize(teacher_img.detach(), dim=-1)
                student_keep = student_cam[keep]
                loss_cam = 1.0 - torch.nn.functional.cosine_similarity(
                    student_keep, teacher_img, dim=-1
                ).mean()

        loss_ssl = self.lidar_loss_weight * loss_lidar + self.cam_loss_weight * loss_cam

        return dict(
            loss=loss_ssl,
            loss_ssl=loss_ssl,
            loss_lidar=loss_lidar,
            loss_cam=loss_cam,
            matched_points=matched_points,
        )
@MODELS.register_module()
class MOCHAPretrainerV3NOMAELite(MOCHAPretrainerV3Masked):
    """
    MOCHA-v3 NOMAE-lite:
    EMA LiDAR self-distillation plus lightweight multi-scale
    neighborhood occupancy prediction from LiDAR features.
    """

    def __init__(
        self,
        *args,
        occ_loss_weight=0.15,
        occ_scales=(0.4, 0.8, 1.6),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.occ_loss_weight = occ_loss_weight
        self.occ_scales = list(occ_scales)
        self.occ_head = nn.Sequential(
            nn.Linear(self.student_proj[0].in_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, len(self.occ_scales)),
        )

    def _make_batch(self, input_dict, n):
        if "batch" in input_dict:
            return input_dict["batch"][:n].long()

        if "offset" in input_dict:
            offset = input_dict["offset"].long()
            batch = torch.empty(n, device=offset.device, dtype=torch.long)
            start = 0
            for i, end in enumerate(offset):
                end_i = min(int(end.item()), n)
                if end_i > start:
                    batch[start:end_i] = i
                start = end_i
            return batch

        return torch.zeros(n, device=input_dict["coord"].device, dtype=torch.long)

    def _occupancy_targets(self, input_dict, n):
        coord = input_dict["coord"][:n]
        batch = self._make_batch(input_dict, n)
        targets = []

        for scale in self.occ_scales:
            grid = torch.floor(coord / float(scale)).long()
            key = torch.cat([batch[:, None], grid], dim=1)
            _, inv = torch.unique(key, dim=0, return_inverse=True)
            counts = torch.bincount(inv).float()
            local_count = counts[inv]
            targets.append((local_count > 1).float())

        return torch.stack(targets, dim=1)

    def _occ_loss(self, feat, input_dict):
        n = min(feat.shape[0], input_dict["coord"].shape[0])
        if n <= 1:
            return feat.sum() * 0.0

        logits = self.occ_head(feat[:n])
        target = self._occupancy_targets(input_dict, n).to(logits.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    def forward(self, input_dict, return_point=False):
        if not self.teacher_initialized:
            self._sync_teacher()
        else:
            self._ema_update()

        g1_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.global_coord_jitter,
            color_drop_prob=self.global_color_drop,
        )
        s1_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.strong_coord_jitter,
            color_drop_prob=self.strong_color_drop,
        )
        teacher_input = self._clone_aug_input(
            input_dict,
            coord_jitter_std=self.teacher_coord_jitter,
            color_drop_prob=0.0,
        )

        g1_coarse, g1_final = self._extract_feats(self.backbone, g1_input)
        s1_coarse, _ = self._extract_feats(self.backbone, s1_input)

        with torch.no_grad():
            teacher_coarse, _ = self._extract_feats(
                self.teacher_backbone, teacher_input
            )

        teacher_feat = self._norm_proj(self.teacher_proj, teacher_coarse)

        g_masked = self._mask_feat(g1_coarse, self.global_mask_ratio)
        s_masked = self._mask_feat(s1_coarse, self.strong_mask_ratio)

        views = [
            self._norm_proj(self.student_proj, g1_coarse),
            self._norm_proj(self.student_proj, s1_coarse),
            self._norm_proj(self.student_proj, g_masked),
            self._norm_proj(self.student_proj, s_masked),
        ]

        loss_terms = []
        for v in views:
            loss_terms.append(
                1.0
                - torch.nn.functional.cosine_similarity(
                    v, teacher_feat, dim=-1
                ).mean()
            )
        loss_lidar = torch.stack(loss_terms).mean()

        loss_occ = 0.5 * (
            self._occ_loss(g_masked, input_dict)
            + self._occ_loss(s_masked, input_dict)
        )

        loss_cam = g1_final.sum() * 0.0
        matched_points = torch.zeros((), device=g1_final.device, dtype=torch.long)

        loss_ssl = (
            self.lidar_loss_weight * loss_lidar
            + self.occ_loss_weight * loss_occ
            + self.cam_loss_weight * loss_cam
        )

        return dict(
            loss=loss_ssl,
            loss_ssl=loss_ssl,
            loss_lidar=loss_lidar,
            loss_occ=loss_occ,
            loss_cam=loss_cam,
            matched_points=matched_points,
        )


@MODELS.register_module()
class TargetMOCHASegmentorV2(DefaultSegmentorV2):
    def __init__(
        self,
        num_classes,
        backbone_out_channels,
        backbone=None,
        criteria=None,
        freeze_backbone=False,
        occ_loss_weight=0.02,
        occ_scales=(0.2, 0.4, 0.8),
    ):
        super().__init__(
            num_classes=num_classes,
            backbone_out_channels=backbone_out_channels,
            backbone=backbone,
            criteria=criteria,
            freeze_backbone=freeze_backbone,
        )
        self.occ_loss_weight = occ_loss_weight
        self.occ_scales = list(occ_scales)
        self.occ_head = nn.Sequential(
            nn.Linear(backbone_out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, len(self.occ_scales)),
        )

    def _make_batch(self, input_dict, n):
        if "batch" in input_dict:
            return input_dict["batch"][:n].long()

        if "offset" in input_dict:
            offset = input_dict["offset"].long()
            batch = torch.empty(n, device=offset.device, dtype=torch.long)
            start = 0
            for i, end in enumerate(offset):
                end_i = min(int(end.item()), n)
                if end_i > start:
                    batch[start:end_i] = i
                start = end_i
            return batch

        return torch.zeros(n, device=input_dict["coord"].device, dtype=torch.long)

    def _occupancy_targets(self, input_dict, n):
        coord = input_dict["coord"][:n]
        batch = self._make_batch(input_dict, n)
        targets = []

        for scale in self.occ_scales:
            grid = torch.floor(coord / float(scale)).long()
            key = torch.cat([batch[:, None], grid], dim=1)
            _, inv = torch.unique(key, dim=0, return_inverse=True)
            counts = torch.bincount(inv).float()
            targets.append((counts[inv] > 1).float())

        return torch.stack(targets, dim=1)

    def _occ_loss(self, feat, input_dict):
        n = min(feat.shape[0], input_dict["coord"].shape[0])
        if n <= 1:
            return feat.sum() * 0.0

        logits = self.occ_head(feat[:n])
        target = self._occupancy_targets(input_dict, n).to(logits.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)

    def forward(self, input_dict, return_point=False):
        point = Point(input_dict)
        point = self.backbone(point)

        if isinstance(point, Point):
            while "pooling_parent" in point.keys():
                assert "pooling_inverse" in point.keys()
                parent = point.pop("pooling_parent")
                inverse = point.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
                point = parent
            feat = point.feat
        else:
            feat = point

        seg_logits = self.seg_head(feat)
        return_dict = {}

        if return_point:
            return_dict["point"] = point

        if self.training:
            loss_seg = self.criteria(seg_logits, input_dict["segment"])
            loss_occ = self._occ_loss(feat, input_dict)
            loss = loss_seg + self.occ_loss_weight * loss_occ
            return_dict["loss"] = loss
            return_dict["loss_seg"] = loss_seg
            return_dict["loss_occ"] = loss_occ
        elif "segment" in input_dict.keys():
            loss = self.criteria(seg_logits, input_dict["segment"])
            return_dict["loss"] = loss
            return_dict["seg_logits"] = seg_logits
        else:
            return_dict["seg_logits"] = seg_logits

        return return_dict
