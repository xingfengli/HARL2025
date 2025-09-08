import os
import numpy as np
import scipy.io as sio
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights
from torch.nn import functional as F
# 鸟类种类
bird_classes = [
    "Agelaius phoeniceus", "Cardinalis cardinalis", "Certhia americana",
    "Corvus brachyrhynchos", "Molothrus ater", "Setophaga aestiva",
    "Setophaga ruticilla", "Spinus tristis", "Tringa semipalmata", "Turdus migratorius"
]

# 路径定义

mel_path = r"F:\NF\mel\data_wav_8s_2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 计算 Delta 和 Delta-Delta 特征的函数
def compute_deltas(features):
    delta = np.diff(features, axis=1, append=features[:, -1:])  # 时间方向差分
    delta_delta = np.diff(delta, axis=1, append=delta[:, -1:])  # 再次差分
    return delta, delta_delta


# 数据集加载时的路径修改为 mel 目录
region_paths = {
    'region1': {'mel': os.path.join(mel_path, '1')},  # 修改为 mel
    'region2': {'mel': os.path.join(mel_path, '2')},  # 修改为 mel
    'region3': {'mel': os.path.join(mel_path, '3')}   # 修改为 mel
}

class BirdDataset(Dataset):
    def __init__(self, region_path, bird_classes, transform=None, augmentation=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.augmentation = augmentation

        for label, bird_class in enumerate(bird_classes):
            mel_dir = os.path.join(region_path['mel'], bird_class)  # 修改为 mel

            if not os.path.exists(mel_dir):
                continue

            for file in os.listdir(mel_dir):
                if file.endswith('.mat'):
                    mel_file = os.path.join(mel_dir, file)

                    mel_data = sio.loadmat(mel_file)
                    mel_feature = mel_data.get('melSpecdb')  # 这里根据实际情况修改为 melSpecdb
                    if mel_feature is None:
                        continue

                    mel_delta, mel_delta_delta = compute_deltas(mel_feature)
                    mel_combined = np.concatenate([mel_feature, mel_delta, mel_delta_delta], axis=0)
                    mel_combined = np.expand_dims(mel_combined, axis=0)

                    self.data.append(mel_combined)
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_feature = self.data[idx]
        label = self.labels[idx]

        # 应用增强
        if self.augmentation:
            mel_feature = self.augmentation(mel_feature)

        # 应用变换
        if self.transform:
            mel_feature = self.transform(mel_feature)

        return mel_feature, label


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, mel=2, num_classes=10):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.mel = mel
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.mel * ce_loss
        return loss.mean()

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):
        return self.attention(x, x, x)[0]


# 定义模型
# ResNetBranch with more attention heads and higher embedding dimension
class ResNetBranch(nn.Module):
    def __init__(self, input_channels=1, num_heads=4, embed_dim=2048):
        super(ResNetBranch, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False).to(device)
        self.resnet.fc = nn.Identity()
        self.multihead_attention = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.batch_norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        features = self.resnet(x)
        features = features.unsqueeze(1)
        attn_features = self.multihead_attention(features)
        normed_features = self.batch_norm(attn_features.mean(dim=1))
        return normed_features


class MultiBranchCNN_M_WI(nn.Module):
    def __init__(self, num_classes):
        super(MultiBranchCNN_M_WI, self).__init__()
        self.mel_branch = ResNetBranch(input_channels=1)

        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, mel):

        mel_features = self.mel_branch(mel)


        return self.fc(mel_features)

# 数据集加载和划分
def load_data(region_paths, bird_classes):
    datasets = {region: BirdDataset(region_path=paths, bird_classes=bird_classes)
                for region, paths in region_paths.items()}
    data_loaders = {}

    train_dataset = datasets['region1']
    test_dataset_region2 = datasets['region2']
    test_dataset_region3 = datasets[('region3')]

    data_loaders['train'] = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    data_loaders['test_region2'] = DataLoader(test_dataset_region2, batch_size=32, shuffle=False, drop_last=False)
    data_loaders['test_region3'] = DataLoader(test_dataset_region3, batch_size=32, shuffle=False, drop_last=False)

    return data_loaders

# 区域路径配置
region_paths = {
    'region1': {'mel': os.path.join(mel_path, '1')},
    'region2': {'mel': os.path.join(mel_path, '2')},
    'region3': {'mel': os.path.join(mel_path, '3')}
}

# 模型训练函数
def train_model(model, criterion, optimizer, train_loader, num_epochs=40, save_epoch=None):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    best_model_wts = None
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            del inputs, labels, outputs, preds
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 更新学习率调度器
        scheduler.step(epoch_loss)

        # 保存当前最优模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()

    if save_epoch is not None and epoch + 1 == save_epoch:
        torch.save(model.state_dict(), f"model_epoch_{save_epoch}.pth")
        print(f"Model saved at epoch {save_epoch} to'model_epoch_{save_epoch}.pth'")

    # 加载最优模型权重
    model.load_state_dict(best_model_wts)

    # 保存模型到文件
    torch.save(best_model_wts, "multi_branch_cnn_m_wi_train1_d.pth")
    print("Best model saved to 'best_model.pth'")


# 模型评估函数
def evaluate_model(model, test_loader, region_name=None):
    model.eval()
    all_preds = []
    all_labels = []
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.float().to(device), labels.to(device)  # 确保 labels 在 GPU 上
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            running_corrects += torch.sum(preds == labels.data)
    torch.cuda.empty_cache()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    conf_matrix = confusion_matrix(all_labels, all_preds)
    print(f'Confusion Matrix for {region_name}:\n{conf_matrix}')

    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = running_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy for {region_name}: {accuracy:.4f}, UAR: {recall:.4f}, F1 Score: {f1:.4f}')
if __name__ == '__main__':
    model = MultiBranchCNN_M_WI(num_classes=len(bird_classes)).to(device)
    criterion = FocalLoss(alpha=0.25, mel=2, num_classes=len(bird_classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# 加载数据
    data_loaders = load_data(region_paths, bird_classes)

# 训练模型
    train_model(model, criterion, optimizer, data_loaders['train'], num_epochs=40,save_epoch=40)

# 评估模型
    evaluate_model(model, data_loaders['test_region2'], "Region 2")

