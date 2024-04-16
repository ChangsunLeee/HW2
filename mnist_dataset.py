import os
import torch
from torchvision.datasets import DatasetFolder
from torchvision import transforms
from PIL import Image
import tarfile

class MNIST(torch.utils.data.Dataset):
    def __init__(self, data_file):
        self.data_file = data_file
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dataset = DatasetFolder(self.extract_tar(data_file), loader=self.loader, extensions=('png',), transform=self.transform)

    def extract_tar(self, data_file):
        data_dir = os.path.splitext(data_file)[0]
        if not os.path.isdir(data_dir):
            with tarfile.open(data_file, 'r') as tar:
                tar.extractall(data_dir)
        return data_dir

    def loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        label = int(os.path.splitext(os.path.basename(self.dataset.samples[idx][0]))[0].split('_')[1])
        return image, label

# 예시 코드 시작점
if __name__ == "__main__":
    # 데이터 파일 경로 (상대 경로로 변경)
    train_tar_path = "./train.tar"
    test_tar_path = "./test.tar"

    # MNIST 데이터셋 로드 (data_file 전달)
    train_dataset = MNIST("train.tar")
    test_dataset = MNIST("test.tar")

    # 데이터셋 크기 확인
    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
