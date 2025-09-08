import torch
import numpy as np

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from gamma_mel_res50_multiat_d import device, MultiBranchCNN_G_M_WI, bird_classes, BirdDataset, region_paths


def plot_confusion_matrix_with_legend(model, data_loader, bird_classes, title, name):
    """
    使用保存的模型生成并绘制混淆矩阵，简化轴标签为数字，并提供旁边的对照表。

    Args:
        model: 训练好的模型。
        data_loader: 测试数据的 DataLoader。
        bird_classes: 鸟类种类列表。
        title: 图的标题。
        name: 保存的图片文件名。
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_corrects = 0
    with torch.no_grad():
        for (gamma, mel), labels in data_loader:
            gamma, mel, labels = gamma.float().to(device), mel.float().to(device), labels.to(device)
            outputs = model(gamma, mel)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            running_corrects += torch.sum(preds == labels.data)

    # 展平预测值和真实值
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = running_corrects.double() / len(data_loader.dataset)
    print(f'Test Accuracy for : {accuracy:.4f}, UAR: {recall:.4f}, F1 Score: {f1:.4f}')

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))  # 调整图形大小


    # 绘制混淆矩阵（数字作为标签）
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(len(bird_classes)))
    colorbar = disp.plot(cmap="YlGn", ax=ax, colorbar=True).im_.colorbar

    colorbar.ax.yaxis.set_tick_params(labelsize=20)  # 刻度字体大小
    colorbar.ax.yaxis.set_tick_params(labelsize=20)  # 刻度字体大小

    colorbar.ax.tick_params(labelsize=20)  # 刻度值字体大小


    for text in disp.text_.ravel():  # `disp.text_` 包含所有内部的文本对象
        text.set_fontsize(20)  # 设置内部字体大小
        text.set_fontweight('bold')  # 设置内部字体加粗
        # text.set_fontname('Times New Roman')
        # 设置标题和轴标签



    ax.set_title(title, fontsize=24, fontname='Times New Roman', fontweight='bold',pad=15)  # 加粗标题
    ax.set_xlabel("Predicted Label", fontsize=22, fontname='Times New Roman', fontweight='bold')  # 加粗x轴
    ax.set_ylabel("True Label", fontsize=22, fontname='Times New Roman', fontweight='bold')  # 加粗y轴



    # 调整刻度字体大小和样式
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.set_tick_params(labelsize=22)
    ax.yaxis.set_tick_params(labelsize=22)



    # 调整布局，去除空白
    plt.tight_layout()

    # 保存为高分辨率 SVG 图片
    fig.savefig(name, format='png', dpi=300,bbox_inches='tight')
    plt.show()


# 示例用法
if __name__ == "__main__":
    # 加载模型
    model = MultiBranchCNN_G_M_WI(num_classes=len(bird_classes)).to(device)
    model.load_state_dict(torch.load("multi_branch_cnn_g_m_wi_train3_d.pth", map_location=device))

    # 准备测试数据集加载器 (如 Region 1)
    test_dataset_region1 = BirdDataset(region_path=region_paths['region1'], bird_classes=bird_classes)
    data_loader_region1 = DataLoader(test_dataset_region1, batch_size=32, shuffle=False,drop_last=False)
    #
    test_dataset_region3 = BirdDataset(region_path=region_paths['region2'], bird_classes=bird_classes)
    data_loader_region3 = DataLoader(test_dataset_region3, batch_size=32, shuffle=False,drop_last=False)

    # 绘制 Region 1 的混淆矩阵
    plot_confusion_matrix_with_legend(model, data_loader_region1, bird_classes, title=r"$D_3D_1$",name="D3D1_nolable.png")
    plot_confusion_matrix_with_legend(model, data_loader_region3, bird_classes, title=r"$D_3D_2$",name="D3D2_nolable.png")
