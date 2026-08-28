Glaucoma Unified Training Pipeline

This repository contains code for a python package used for the automated training and evaluation of CNN models on Glaucoma Datasets.

The package supports the following Datasets:
1. Drishti-GS.
2. ORIGA
3. REFUGE

The pipeline is designed so that each training & evaluation run is highly configurable between the following parameters:


Examine run_builder.py for more options.

Example Code:
The below code trains EfficientUnet-b7++ with specified training and validation transforms, specified visualization samples, data augmentation x4, 100 epochs

train_transforms = [
    {"type": "Resize", "height": 512, "width": 512},
    {"type": "HorizontalFlip", "p": 0.5},
    {"type": "VerticalFlip", "p": 0.5},
    {"type":"OneOf",'transforms':[
        {"type": "GridDistortion", 'num_steps':5, "distort_limit":0.3, "border_mode":cv2.BORDER_CONSTANT, "mask_value":0, "p":1},
        {"type": "OpticalDistortion", "distort_limit":0.1, "shift_limit":0.1, "border_mode":cv2.BORDER_CONSTANT, "mask_value":0, "p":1},
    ], "p":0.7},
    {"type": "ShiftScaleRotate", "scale_limit":0.2, "rotate_limit":30, "shift_limit":0.1, "border_mode":cv2.BORDER_CONSTANT, "mask_value":0, "p":0.5},
    {"type": "RandomBrightnessContrast", "brightness_limit":0.2, "contrast_limit":0.2, "p":0.5},
    {"type": "RandomGamma", "gamma_limit":(80, 120), "p":0.5},
    #{"type": "GaussNoise", "var_limit":(10.0, 50.0), "p":0.3},
    {"type": "CLAHE", "clip_limit": 8.0, "tile_grid_size": (8, 8)},
    {"type": "Normalize", "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    {"type": "ToTensorV2"},
]

val_transforms = [
    {"type": "Resize", "height": 512, "width": 512},
    {"type": "CLAHE", "clip_limit": 8.0, "tile_grid_size": (8, 8)},
    {"type": "Normalize", "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
    {"type": "ToTensorV2"}
    ]
config = RunConfig(
    device=DEVICE,
    epochs=100,
    img_height=512,
    img_width=512,
    accumulation_steps=1,
    model=ModelConfig(type="efficientunet-b7pp", kwargs={'encoder_weights': 'imagenet', 'in_channels':3, 'classes':3}),
    batch_size=20,
    loss=LossConfig(losses=[
        SingleLossConfig(type="CE", weight=0.5, kwargs={}),
        SingleLossConfig(type="DICE", weight=0.5, kwargs={"mode": "multiclass", 'classes': [1,2]})
    ]),
    optimizer=OptimizerConfig(type="ADAM", lr=0.0002),
    scheduler=SchedulerConfig(type="CosineAnnealingLR", kwargs={"T_max": 100}),
    scaler=ScalerConfig(enabled=True),
    dataset=DatasetConfig(
        name="REFUGE",
        roi=False,
        local_root="datasets/REFUGE",
        drive_root="/content/drive/MyDrive/CDR Paper/datasets/REFUGE",
        augmentation=AugmentationConfig(
            train_transforms=train_transforms,
            val_transforms=val_transforms,
            multiplier=4,
        ),
        in_memory=False,
    ),
    root_exp_dir="runs",
    exp_title_postfix=f'seed-{i}',
    num_workers=4,
    #visualization_samples=[1,2]
    visualization_samples=['V0016.jpg','V0027.jpg','V0029.jpg','V0034.jpg','V0052.jpg','V0057.jpg','V0078.jpg','V0079.jpg','V0082.jpg','V0097.jpg','V0141.jpg','V0178.jpg','V0182.jpg','V0193.jpg','V0206.jpg','V0256.jpg','V0328.jpg','V0332.jpg','V0380.jpg','V0390.jpg'],
    visualization_epochs=[1, 10, 15, 20, 25, 30, 35, 40, 45 ,50, 60, 70, 80, 90, 95, 100],
    #visualization_epochs=[1, 2, 3, 4, 5],
)

run = RunBuilder().build(config)
run.execute()
