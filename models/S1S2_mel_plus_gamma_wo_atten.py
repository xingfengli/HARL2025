import os
import numpy as np
import scipy.io as sio
import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, recall_score, f1_score
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 鸟类种类
bird_classes = [
    "AMCR", "AMRO", "BLJA",
    "ECMC", "ECMK", "ENGE",
    "FFCR", "RNFL"
]

# 数据路径
gamma_path = "/root/autodl-tmp/gamma/DB2"
mel_path = "/root/autodl-tmp/mel/DB2"

# 计算 delta 和 delta-delta
def compute_deltas(features):
    delta = np.diff(features, axis=1, append=features[:, -1:])
    delta_delta = np.diff(delta, axis=1, append=delta[:, -1:])
    return delta, delta_delta

# 数据集类
class BirdDataset(Dataset):
    def __init__(self, region_path, bird_classes, transform=None, augmentation=None):
        self.data = []
        self.labels = []
        self.transform = transform
        self.augmentation = augmentation

        for label, bird_class in enumerate(bird_classes):
            gamma_dir = os.path.join(region_path['gamma'], bird_class)
            mel_dir = os.path.join(region_path['mel'], bird_class)
            if not os.path.exists(gamma_dir) or not os.path.exists(mel_dir):
                continue

            for file in os.listdir(gamma_dir):
                if file.endswith('.mat'):
                    gamma_file = os.path.join(gamma_dir, file)
                    mel_file = os.path.join(mel_dir, file)

                    # 加载 Gamma 特征
                    gamma_data = sio.loadmat(gamma_file)
                    gamma_feature = gamma_data.get('gammaSpecdb')
                    if gamma_feature is None:
                        continue
                    gamma_delta, gamma_delta_delta = compute_deltas(gamma_feature)
                    gamma_combined = np.concatenate([gamma_feature, gamma_delta, gamma_delta_delta], axis=0)
                    gamma_combined = np.expand_dims(gamma_combined, axis=0)

                    # 加载 Mel 特征
                    mel_data = sio.loadmat(mel_file)
                    mel_feature = mel_data.get('melSpecdb')
                    if mel_feature is None:
                        continue
                    mel_delta, mel_delta_delta = compute_deltas(mel_feature)
                    mel_combined = np.concatenate([mel_feature, mel_delta, mel_delta_delta], axis=0)
                    mel_combined = np.expand_dims(mel_combined, axis=0)

                    self.data.append((gamma_combined, mel_combined))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gamma_feature, mel_feature = self.data[idx]
        label = self.labels[idx]
        if self.augmentation:
            gamma_feature = self.augmentation(gamma_feature)
            mel_feature = self.augmentation(mel_feature)
        if self.transform:
            gamma_feature = self.transform(gamma_feature)
            mel_feature = self.transform(mel_feature)
        return (gamma_feature, mel_feature), label

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return loss.mean()

# ResNet 分支（无注意力）
class ResNetBranch(nn.Module):
    def __init__(self, input_channels=1):
        super(ResNetBranch, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        features = self.resnet(x)
        return features

# 多分支网络
class MultiBranchCNN_G_M_WO(nn.Module):
    def __init__(self, num_classes):
        super(MultiBranchCNN_G_M_WO, self).__init__()
        self.mel_branch = ResNetBranch(input_channels=1)
        self.gamma_branch = ResNetBranch(input_channels=1)
        self.fc = nn.Sequential(
            nn.Linear(2048*2, 256),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(64, num_classes)
        )

    def forward(self, gamma, mel):
        mel_features = self.mel_branch(mel)
        gamma_features = self.gamma_branch(gamma)
        combined_features = torch.cat((gamma_features, mel_features), dim=1)
        return self.fc(combined_features)

# 数据加载
def load_data(region_paths, bird_classes):
    datasets = {region: BirdDataset(region_path=paths, bird_classes=bird_classes)
                for region, paths in region_paths.items()}
    data_loaders = {
        'train': DataLoader(datasets['region1'], batch_size=16, shuffle=True, drop_last=True),
        'test_region2': DataLoader(datasets['region2'], batch_size=32, shuffle=False, drop_last=False)
    }
    return data_loaders

# 评估函数
def evaluate_model(model, test_loader, region_name=None):
    model.eval()
    all_preds, all_labels = [], []
    running_corrects = 0
    with torch.no_grad():
        for (gamma, mel), labels in test_loader:
            gamma = gamma.float().to(device)
            mel = mel.float().to(device)
            labels = labels.to(device)
            outputs = model(gamma, mel)
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
    print(f'Accuracy: {accuracy:.4f}, UAR: {recall:.4f}, F1 Score: {f1:.4f}')
    return f1

# 训练函数
def train_model(model, criterion, optimizer, train_loader, val_loader=None, num_epochs=20):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    best_model_wts = None
    best_f1 = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for (gamma, mel), labels in train_loader:
            gamma = gamma.float().to(device)
            mel = mel.float().to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(gamma, mel)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * gamma.size(0)
            running_corrects += torch.sum(preds == labels.data)
            del gamma, mel, labels, outputs, preds
            torch.cuda.empty_cache()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        f1_r2 = 0.0
        if val_loader is not None:
            f1_r2 = evaluate_model(model, val_loader, region_name="Validation Region 2")

        if f1_r2 > best_f1:
            best_f1 = f1_r2
            best_model_wts = model.state_dict()
            torch.save(best_model_wts, "best_s1s2_mel_gamma_wo_atten_f1.pth")
            print(f"New best model saved! F1={f1_r2:.4f}")

        scheduler.step(epoch_loss)

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        print(f"Training finished. Best model saved as 'best_s1s2_mel_gamma_wo_atten_f1.pth' with F1={best_f1:.4f}")

# 主程序
if __name__ == '__main__':
    model = MultiBranchCNN_G_M_WO(num_classes=len(bird_classes)).to(device)
    criterion = FocalLoss(alpha=0.25, gamma=1.5, num_classes=len(bird_classes)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-3)

    region_paths = {
        'region1': {'gamma': os.path.join(gamma_path, 'S1'), 'mel': os.path.join(mel_path, 'S1')},
        'region2': {'gamma': os.path.join(gamma_path, 'S2'), 'mel': os.path.join(mel_path, 'S2')}
    }

    data_loaders = load_data(region_paths, bird_classes)

    # 训练模型
    train_model(model, criterion, optimizer, 
                train_loader=data_loaders['train'],
                val_loader=data_loaders['test_region2'],
                num_epochs=40)

    # 加载最佳模型进行最终测试
    print("\n=== Loading Best Model for Final Testing ===")
    best_model_path = "best_s1s2_mel_gamma_wo_atten_f1.pth"
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    print("\n=== Testing on Region 2 ===")
    evaluate_model(model, data_loaders['test_region2'], "Region 2")
