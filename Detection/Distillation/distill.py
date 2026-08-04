import lightly_train

if __name__ == "__main__":
    lightly_train.pretrain(
        out="out/distill_ViTl_yolo26n",
        resume_interrupted=True,
        data="/home/ross/datasets/yolo11_Training/train/images",
        model="ultralytics/yolo26n.yaml",
        method="distillationv3",
        method_args={
            "teacher": "dinov3/vitl16",
            "teacher_args": {"pretrained": False},
            "teacher_weights": "/home/ross/DinoV3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            "temperature_global": 0.1,
            "temperature_local": 0.1,
        }
    )