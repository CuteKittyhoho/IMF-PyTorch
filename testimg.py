import pickle
import numpy as np
import torch
import torch.nn as nn

# 加载现有的 pkl 文件
with open(r'datasets/DB15K/img_features.pkl', 'rb') as f:
    embeddings = pickle.load(f)  # 假设维度为 (14951, 4096)

# 将 embeddings 转换为 NumPy 数组，然后转换为 PyTorch 张量
embeddings = np.array(embeddings)
embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)  # (14951, 4096)

# 定义线性层用于降维，将 4096 维转换为 1000 维
linear_layer = nn.Linear(4096, 1000)

# 不需要计算梯度
with torch.no_grad():
    reduced_embeddings = linear_layer(embeddings_tensor)  # (14951, 1000)

# 转回 NumPy 数组
reduced_embeddings = reduced_embeddings.numpy()  # (14951, 1000)

# 将维度扩展为 (14951, 64, 1000)
expanded_embeddings = np.repeat(reduced_embeddings[:, np.newaxis, :], 64, axis=1)

# 保存扩展后的 pkl 文件
output_path = r'datasets/DB15K/d_img.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(expanded_embeddings, f)

print("转换完成，保存的文件路径为:", output_path)







