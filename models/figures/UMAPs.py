import torch
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.sandbox.distributions.examples.ex_mvelliptical import fig
from torch.utils.data import DataLoader

import umap
from sklearn.preprocessing import LabelEncoder

from gamma_mel_res50_d import MultiBranchCNN_G_M_WO
from gamma_mel_res50_multiat_d import BirdDataset, region_paths, bird_classes
from gamma_mel_res50_multiat_d import MultiBranchCNN_G_M_WI
from mel_res50_mult_d import MultiBranchCNN_M_WI


# 获取模型特征并提取数据
def extract_features(model, data_loader, device):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for (gamma,mel), label in data_loader:
            gamma,mel = gamma.float().to(device),mel.float().to(device)
            label = label.to(device)
            outputs = model( gamma,mel)
            features.append(outputs.cpu().numpy())
            labels.append(label.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import numpy as np

import umap
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def plot_umap(features, labels, bird_classes,name, title='Train2-Test(w/i)'):
    """
    绘制UMAP图，使用自定义颜色映射，并将图例移到坐标轴外。

    Args:
        features: 输入特征数据。
        labels: 类别标签。
        bird_classes: 鸟类类别名称列表。
        title: 图的标题。
    """
    # 使用UMAP进行降维
    umap_model = umap.UMAP(n_components=2,
                           n_neighbors=30,  # 增加邻居数，使局部结构更加平滑
                           min_dist=0.9,  # 增大min_dist，增加类之间的间隔
                           random_state=42)
    umap_features = umap_model.fit_transform(features)

    # 生成颜色映射
    custom_colors = [
        (153 / 255, 229 / 255, 255 / 255),  # 0 Agelaius phoeniceus
        (242 / 255, 170 / 255, 132 / 255),  # 1 Cardinalis cardinalis
        (35 / 255, 61 / 255, 220 / 255),  # 2 Certhia americana
        (25 / 255, 101 / 255, 41 / 255),  # 3 Corvus brachyrhynchos
        (255 / 255, 140 / 255, 0 / 255),  # 4 Molothrus ater
        (98 / 255, 192 / 255, 171 / 255),  # 5 Setophaga aestiva
        (115 / 255, 237 / 255, 0 / 255),  # 6 Setophaga ruticilla
        (101 / 255, 103 / 255, 50 / 255),  # 7 Spinus tristis
        (228 / 255, 0 / 255, 232 / 255),  # 8 Tringa semipalmata
        (254 / 255, 255 / 255, 0 / 255)  # 9 Turdus migratorius
    ]

    # 使用 LabelEncoder 对标签编码
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # 绘制UMAP图
    plt.figure(figsize=(10, 8))
    for i, bird_class in enumerate(bird_classes):
        class_mask = (labels_encoded == i)
        plt.scatter(umap_features[class_mask, 0],
                    umap_features[class_mask, 1],
                    color=custom_colors[i],
                    label=f"{i}: {bird_class}",
                    s=20,
                    alpha=0.7)

    # 调整图例位置到图的外部
    # plt.legend(
    #
    #     loc='upper right',  # 将图例放在左上角
    #     # fontsize=18,  # 调整图例字体大小
    #     title_fontsize=20,  # 调整图例标题字体大小
    #     bbox_to_anchor=(1, 1),  # 设置图例相对于坐标轴的位置
    #     framealpha=0.3,
    #     prop={'family': 'Times New Roman', 'size': 27}  # 设置图例的字体
    # )



    # 自定义刻度仅显示 0 和 10 的倍数
    x_ticks = np.arange(0, np.ceil(umap_features[:, 0].max() / 10) * 10 + 1, 10)
    y_ticks = np.arange(0, np.ceil(umap_features[:, 1].max() / 10) * 10 + 1, 10)
    plt.xticks(ticks=x_ticks, fontsize=30, fontname='Times New Roman')  # 调整 x 轴刻度字体
    plt.yticks(ticks=y_ticks, fontsize=30, fontname='Times New Roman')  # 调整 y 轴刻度字体

    # 设置标题和显示
    plt.title(title, fontsize=30, fontweight='bold', fontname='Times New Roman',pad=15)  # 增大标题字体
    plt.tight_layout()  # 自动调整布局，确保内容不被裁剪
    plt.savefig(name, format='png', dpi=300)
    plt.show()


# 可视化 UMAP

# 加载数据并提取特征
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载已训练好的模型
model = MultiBranchCNN_G_M_WI(num_classes=10).to(device)
model.load_state_dict(torch.load("multi_branch_cnn_g_m_wi_train1_d.pth", map_location=device))

# 加载数据集
test_dataset_region1 = BirdDataset(region_path=region_paths['region3'], bird_classes=bird_classes)
test_loader_region1 = DataLoader(test_dataset_region1, batch_size=32, shuffle=False, drop_last=False)

test_dataset_region3 = BirdDataset(region_path=region_paths['region2'], bird_classes=bird_classes)
test_loader_region3 = DataLoader(test_dataset_region3, batch_size=32, shuffle=False, drop_last=False)

# # 提取 Region1 和 Region3 的特征
features_region1, labels_region1 = extract_features(model, test_loader_region1, device)
features_region3, labels_region3 = extract_features(model, test_loader_region3, device)

# 使用 LabelEncoder 将标签编码为数值
label_encoder = LabelEncoder()
label_encoder.fit(bird_classes)

plot_umap(features_region1, labels_region1, bird_classes, r"D1D3 w_i MHA.png",r"$D_1D_3$ w/i MHA")
plot_umap(features_region3, labels_region3, bird_classes, r"D1D2 w_i MHA.png",r"$D_1D_2$ w/i MHA")
