import os
import numpy as np

import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize

from sklearn.metrics import confusion_matrix, recall_score, f1_score, auc, roc_curve
import torch.nn as nn

import gamma_mel_res50_d

import gamma_res50_d

import mel_res50_d

from gamma_mel_res50_d import MultiBranchCNN_G_M_WO
from gamma_mel_res50_multiat_d import MultiBranchCNN_G_M_WI
from gamma_res50_d import MultiBranchCNN_G_WO
from gamma_res50_muti_d import MultiBranchCNN_G_WI
from mel_res50_d import MultiBranchCNN_M_WO
from mel_res50_mult_d import MultiBranchCNN_M_WI

# 路径定义
# 鸟类种类
bird_classes = [
    "Agelaius phoeniceus", "Cardinalis cardinalis", "Certhia americana",
    "Corvus brachyrhynchos", "Molothrus ater", "Setophaga aestiva",
    "Setophaga ruticilla", "Spinus tristis", "Tringa semipalmata", "Turdus migratorius"
]

# 路径定义
gamma_path = r"F:\NF\gamma\data_wav_8s_2"
mel_path = r"F:\NF\mel\data_wav_8s_2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# 区域路径配置
region_paths = {
    'region1': {'gamma': os.path.join(gamma_path, '1'),
                'mel': os.path.join(mel_path, '1')},
    'region2': {'gamma': os.path.join(gamma_path, '2'),
                'mel': os.path.join(mel_path, '2')},
    'region3': {'gamma': os.path.join(gamma_path, '3'),
                'mel': os.path.join(mel_path, '3')}
}
#
# 模型训练函数



model_paths = {
    "Mel+Gamma w/o atten": "multi_branch_cnn_g_m_wo_train1_d.pth",
    "Mel+Gamma w/i atten": "multi_branch_cnn_g_m_wi_train1_d.pth",
    "Mel w/o atten": "multi_branch_cnn_m_wo_train1_d.pth",
    "Mel w/i atten": "multi_branch_cnn_m_wi_train1_d.pth",
    "Gamma w/o atten": "multi_branch_cnn_g_wo_train1_d.pth",
    "Gamma w/i atten": "multi_branch_cnn_g_wi_train1_d.pth"
}

# 定义模型架构对应的类
model_classes = {
    "Mel+Gamma w/o atten": MultiBranchCNN_G_M_WO,
    "Mel+Gamma w/i atten": MultiBranchCNN_G_M_WI,
    "Mel w/o atten": MultiBranchCNN_M_WO,
    "Mel w/i atten": MultiBranchCNN_M_WI,
    "Gamma w/o atten": MultiBranchCNN_G_WO,
    "Gamma w/i atten": MultiBranchCNN_G_WI
}

# 加载已保存的模型到设备
models = {}
for model_name, model_class in model_classes.items():
    # 初始化模型
    model = model_class(len(bird_classes))
    # 加载权重
    checkpoint_path = model_paths.get(model_name)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    # 移动到设备
    models[model_name] = model.to(device)




def evaluate_model_with_probabilities(model_name, model, test_loader, num_classes, region_name=None):
    model.eval()



    all_probs = []
    all_labels = []
    print("Evaluating:", model_name)

    for batch in test_loader:
        if "Mel+Gamma" in model_name:  # 如果模型名称包含 "Mel+Gamma"
            if isinstance(batch, list) or isinstance(batch, tuple):  # 确保 batch 是可解包的类型
                (gamma, mel), labels = batch
                gamma = gamma.float().to(device)
                mel = mel.float().to(device)
        else:  # 单模态模型
            data, labels = batch
            data = torch.stack(data).float().to(device) if isinstance(data, list) else data.float().to(device)

        labels = labels.to(device)

        # 根据模型类型调用对应的 forward 方法
        if "Mel+Gamma" in model_name:
            outputs = model(gamma, mel)
        elif "Mel" in model_name:
            outputs = model(data)
        elif "Gamma" in model_name:
            outputs = model(data)
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        # 转为概率
        softmax = nn.Softmax(dim=1)
        output_prob = softmax(outputs)
        # probs = F.softmax(outputs, dim=1)
        all_labels.extend(labels.tolist())
        all_probs.extend(output_prob.detach().cpu().numpy())
        # print(output_prob.shape)

    # 将结果合并
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    torch.cuda.empty_cache()

    # 如果需要计算 ROC 和 AUC，则对标签进行 One-Hot 编码
    y_true_bin = label_binarize(all_labels, classes=np.arange(len(bird_classes)))

    fpr, tpr, thresholds = roc_curve(y_true_bin.ravel(), all_probs.ravel())
    roc_auc = auc(fpr, tpr)
    # 计算宏平均 ROC 和 AUC


    print(f'ROC AUC for {region_name}: {roc_auc}')
    return fpr, tpr, roc_auc


# 绘制ROC曲线
def plot_roc(results, region):
        plt.figure(figsize=(12, 8))

        for name, data in results.items():
            fpr, tpr, roc_auc = data[region]
            plt.plot(
                fpr, tpr,

                markevery=0.1,
                linewidth=2.5,
                label=f'{name} (AUC = {roc_auc:.2f})',

            )
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {region}')
        plt.legend(loc="lower right")
        plt.show()



# 模型评估函数


if __name__ == '__main__':


    data_loaders = gamma_mel_res50_d.load_data(region_paths, bird_classes)
    data_loaders1 =mel_res50_d.load_data(region_paths, bird_classes)
    data_loaders2 = gamma_res50_d.load_data(region_paths, bird_classes)

    def get_data_loader(model_name,data_loaders1,data_loaders2,data_loaders):
        """
        根据模型名称返回对应的数据加载器。
        """
        print("1",model_name)
        if "Mel+Gamma" in model_name:
            return data_loaders
        if "Mel" in model_name and "Gamma" not in model_name:
            return data_loaders1  # 对应 mel_res50_d 的数据加载器
        elif "Gamma" in model_name and "Mel" not in model_name:
            return data_loaders2  # 对应 gamma_res50_d 的数据加载器


    # 评估模型并动态选择数据加载器
    results = {}
    for name, model in models.items():
        # 动态选择 data_loader
        print(name)

        selected_loader = get_data_loader(name, data_loaders1,data_loaders2,data_loaders)
        # print(selected_loader['test_region1'])  # 查看有哪些键

        print(f"Evaluating {name} on Region 2...")
        fpr1, tpr1, roc_auc1 = evaluate_model_with_probabilities(
            name, model, selected_loader['test_region2'], num_classes=len(bird_classes), region_name=f"{name} Region 2"
        )

        print(f"Evaluating {name} on Region 3...")
        fpr3, tpr3, roc_auc3 = evaluate_model_with_probabilities(
            name, model, selected_loader['test_region3'], num_classes=len(bird_classes), region_name=f"{name} Region 3"
        )

        results[name] = {"Region 2": (fpr1, tpr1, roc_auc1), "Region 3": (fpr3, tpr3, roc_auc3)}
    torch.cuda.empty_cache()
    # 绘制ROC曲线
    plot_roc(results, "Region 2")
    plot_roc(results, "Region 3")
