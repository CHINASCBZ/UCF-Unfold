import torch

old_path = "weights/Italy/best_train_loss.pth"
new_path = "weights/Italy/best.pth"

# 仅加载自己训练或可信来源的 checkpoint
checkpoint = torch.load(
    old_path,
    map_location="cpu",
    weights_only=False
)

# 仅提取模型参数
model_weights = checkpoint["model_state"]

# 保存纯模型权重
torch.save(model_weights, new_path)

print("已保存纯模型权重:", new_path)