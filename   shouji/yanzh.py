from ultralytics import YOLO

model = YOLO('/home/suchang/yolov5-7.0/runs/train/v11_cbam_004/weights/best.pt')
results = model.predict(
    source='/home/suchang/yolov5-7.0/sanghuang/class_v8_003/val/images/ch01_20250822091238_timingCap_aug1.jpg',
    conf=0.25,
    save=True,
    project='/tmp',
    name='test'
)
# 看 /tmp/test/ 里的图有没有框
for r in results:
    print(f'检测到 {len(r.boxes)} 个目标')
    if len(r.boxes) > 0:
        print(f'置信度: {r.boxes.conf.tolist()}')
