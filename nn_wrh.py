import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from tqdm import tqdm

# config
ROOT_DIR = '/home1/gongwk/adni/T1/t1'
SAVE_DIR = '/home1/yangchang/WRH_lab'
BATCH_SIZE = 1
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {"CN": 0, "MCI": 1, "Dementia": 2}

# 数据集
class MRIDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for subject in os.listdir(root_dir):
            subject_path = os.path.join(root_dir, subject)
            if not os.path.isdir(subject_path):
                continue
            for item in os.listdir(subject_path):
                if item.endswith('.anat'):
                    match = re.match(r'.+-(\d+\.?\d*)-(Male|Female)-(CN|MCI|Dementia)-', item)
                    if match:
                        age = float(match.group(1))
                        gender = 1 if match.group(2) == 'Male' else 0
                        label = LABEL_MAP[match.group(3)]
                        anat_dir = os.path.join(subject_path, item)
                        f1 = os.path.join(anat_dir, "T1_brain_1mm_stdspace_linear.nii.gz")
                        f2 = os.path.join(anat_dir, "T1_brain_1mm_stdspace.nii.gz")
                        f3 = os.path.join(anat_dir, "T1_brain_2mm_stdspace.nii.gz")
                        if not all(os.path.exists(f) for f in [f1, f2, f3]):
                            print(f"Skipping incomplete folder: {anat_dir}")
                            continue
                        self.samples.append((f1, f2, f3, age, gender, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f1, f2, f3, age, gender, label = self.samples[idx]
        x1 = nib.load(f1).get_fdata().astype(np.float32)
        x2 = nib.load(f2).get_fdata().astype(np.float32)
        x3 = nib.load(f3).get_fdata().astype(np.float32)

        # 将 x2, x3 resize 到与 x1 相同大小
        target_shape = x1.shape
        x2 = zoom(x2, np.array(target_shape) / np.array(x2.shape), order=1)
        x3 = zoom(x3, np.array(target_shape) / np.array(x3.shape), order=1)

        # 堆叠为 3 通道
        x = np.stack([x1, x2, x3], axis=0)  # shape: [3, D, H, W]
        x = torch.tensor(x)

        age_gender = torch.tensor([age / 100.0, gender], dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return x, age_gender, label

# 网络结构
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x, age_gender):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, age_gender], dim=1)
        return self.fc(x)

# 可视化
def plot_and_save(train_loss, train_acc, val_acc):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "training_plot.png"))

# 测试
def evaluate(model, test_loader):
    model.eval()
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for inputs, age_gender, labels in test_loader:
            inputs, age_gender, labels = inputs.to(DEVICE), age_gender.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs, age_gender)
            _, preds = torch.max(outputs, 1)
            correct_test += (preds == labels).sum().item()
            total_test += labels.size(0)
    acc = correct_test / total_test
    print(f"Test Accuracy: {acc:.4f}")
    with open(os.path.join(SAVE_DIR, "test_accuracy.txt"), 'w') as f:
        f.write(f"Test Accuracy: {acc:.4f}\n")

# 训练流程
def train():
    print("Preparing data...")
    dataset = MRIDataset(ROOT_DIR)
    train_val_set, test_set = train_test_split(dataset, test_size=0.1, random_state=42)
    train_set, val_set = train_test_split(train_val_set, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    train_loss_list, train_acc_list, val_acc_list = [], [], []

    print("Start training!")
    for epoch in range(EPOCHS):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, age_gender, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            inputs, age_gender, labels = inputs.to(DEVICE), age_gender.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs, age_gender)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss_list.append(running_loss / len(train_loader))
        train_acc_list.append(correct / total)

        model.eval()
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for inputs, age_gender, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                inputs, age_gender, labels = inputs.to(DEVICE), age_gender.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs, age_gender)
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
        val_acc_list.append(correct_val / total_val)

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss:.4f}, Train Acc: {correct/total:.4f}, Val Acc: {correct_val/total_val:.4f}")

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))
    plot_and_save(train_loss_list, train_acc_list, val_acc_list)
    evaluate(model, test_loader)

# 主函数
if __name__ == "__main__":
    train()
