import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


test_transform = transforms.Compose([
    transforms.Resize((256,256)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def main():

    test_dataset = datasets.ImageFolder(
            root="picture/test",
            transform=test_transform
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = models.mobilenet_v2(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        1280,
        5
    )

    model.load_state_dict(
        torch.load(
            "models/best_mobilenet_v2.pth"
        )
    )

    model = model.to(device)
    print("模型載入完成 開始測試")
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print(cm)

    accuracy = (
            cm.diagonal().sum()
            /
            cm.sum()
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    class_names = ["Aloe","Zingiber","Mentha","Perilla","Turmeric"]
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig("5.png", dpi=300)
    plt.show()

if __name__=="__main__":
    main()
