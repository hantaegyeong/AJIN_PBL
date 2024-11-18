from ultralytics import YOLO

# YOLOv8 모델 (yolov8s.pt)
model = YOLO('yolov8s.pt')  # yolov8s.pt 가중치 파일 사용

# COCO128 데이터셋으로 학습 - TEST
model.train(data='coco128.yaml', epochs=10, imgsz=640)

# 모델 성능 평가
results = model.val()
print(results)  # 평가 지표 출력
