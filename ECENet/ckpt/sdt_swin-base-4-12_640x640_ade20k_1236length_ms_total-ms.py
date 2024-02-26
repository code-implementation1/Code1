dataset_type = 'ADE20KDataset'
data_root = './ADEChallengeData2016'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=
    '/cache/data/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth',
    backbone=dict(
        # type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=128,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True)),
    decode_head=dict(
        # type='SegDecodingTransformer',
        num_heads=4,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        qkv_bias=True,
        mlp_ratio=4,
        ln_norm_cfg=dict(type='LN', requires_grad=True),
        all_levels=True,
        mask_loss=True,
        ghost_up=True,
        in_channels=(128, 256, 512, 1024),
        channels=256,
        num_classes=150,
        dropout_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        act_cfg=dict(type='ReLU'),
        in_index=(0, 1, 2, 3),
        ratio=1,
        div_loss=True,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            dict(
                type='SDTLoss',
                num_classes=150,
                mask_weight=5.0,
                dice_weight=2.0,
                loss_weight=1.0),
            dict(
                type='DivLoss',
                in_channels=(128, 256, 512, 1024),
                loss_weight=0.2)
        ],
        align_corners=False
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(608, 608)))
work_dir = './work_dirs/sdt_swin-base-4-12_640x640_ade20k_1236length_ms_total'
gpu_ids = range(0, 8)
auto_resume = False
