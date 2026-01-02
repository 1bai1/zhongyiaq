from ultralytics import YOLO

# ==================== 模型选择 ====================
# 选择你想要使用的模型配置文件（取消注释对应的行）

# --- YOLO11 系列 ---
# model = YOLO("ultralytics/cfg/models/11/yolo11.yaml")  # YOLOv11 基础版本

# --- YOLO12 系列---
# model = YOLO("ultralytics/cfg/models/12/yolo12.yaml")  # YOLOv12 基础版本

# --- YOLO13 系列---
model = YOLO("ultralytics/cfg/models/13/yolo13.yaml")  # YOLOv13 基础版本



# ==================== 训练配置 ====================
results = model.train(
    data="dataset/data.yaml",  # 数据集配置文件路径
    epochs=100,                # 训练轮数
    imgsz=640,                 # 输入图像尺寸
    batch=64,                  # 批次大小（根据显存调整）
    name="yolo_training",      # 实验名称（结果保存在 runs/detect/yolo_training）
    # device=3,                # 指定GPU设备（可选，如果不用CUDA_VISIBLE_DEVICES）
    # patience=50,             # 早停耐心值
    # save_period=10,          # 每N个epoch保存一次模型
    # workers=8,               # 数据加载线程数
    # cache=True,              # 缓存图像到内存以加速训练
)

# Print results
print(results)