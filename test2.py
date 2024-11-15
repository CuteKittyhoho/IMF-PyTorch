import pickle
import numpy as np

# 加载现有的 pkl 文件
with open(r'datasets/DB15K/text_features.pkl', 'rb') as f:
    embeddings = pickle.load(f)  # 假设维度为 (14951, 768)

# 将列表转换为NumPy数组
embeddings = np.array(embeddings)

# 将维度扩展为 (14951, 100, 768)
expanded_embeddings = np.repeat(embeddings[:, np.newaxis, :], 100, axis=1)

# 保存扩展后的 pkl 文件
output_path = r'datasets/DB15K/d_text.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(expanded_embeddings, f)

print("转换完成，保存的文件路径为:", output_path)
