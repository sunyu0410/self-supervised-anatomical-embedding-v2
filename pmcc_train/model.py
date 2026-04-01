from sam.models.frameworks.sam import Sam
from mmcv.utils.config import ConfigDict
from mmcv.runner import load_checkpoint

model = Sam(
    backbone={
        "type": "ResNet3d",
        "pretrained2d": True,
        "pretrained": "torchvision://resnet18",
        "depth": 18,
        "in_channels": 1,
        "spatial_strides": (2, 2, 2, 2),
        "temporal_strides": (1, 1, 1, 2),
        "conv1_kernel": (3, 7, 7),
        "conv1_stride_t": 1,
        "conv1_stride_s": 1,
        "pool1_stride_t": 1,
        "with_pool1": False,
        "with_pool2": True,
        "conv_cfg": {"type": "Conv3d"},
        "inflate": ((0, 0), (0, 0), (1, 1), (1, 1)),
        "norm_eval": False,
        "zero_init_residual": False,
    },
    neck={
        "type": "FPN3d",
        "end_level": 3,
        "in_channels": [64, 128, 256],
        "out_channels": 128,
        "num_outs": 3,
        "conv_cfg": {"type": "Conv3d"},
    },
    read_out_head={
        "type": "FPN3d",
        "end_level": 1,
        "in_channels": [512],
        "out_channels": 128,
        "num_outs": 1,
        "conv_cfg": {"type": "Conv3d"},
    },
    # To allow attribute style, wrap train_cfg under ConfigDict
    train_cfg=ConfigDict(
        {
            "pre_select_pos_number": 2000,
            "after_select_pos_number": 100,
            "pre_select_neg_number": 2000,
            "after_select_neg_number": 500,
            "positive_distance": 2.0,
            "ignore_distance": 20.0,
            "coarse_positive_distance": 25.0,
            "coarse_ignore_distance": 5.0,
            "coarse_z_thres": 6.0,
            "coarse_pre_select_neg_number": 250,
            "coarse_after_select_neg_number": 200,
            "coarse_global_select_number": 1000,
            "temperature": 0.5,
        }
    ),
    test_cfg={"save_path": "/data/results/result-dlt/", "output_embedding": True},
)

checkpoint = load_checkpoint(model, "checkpoints/SAM.pth", map_location="cpu")
