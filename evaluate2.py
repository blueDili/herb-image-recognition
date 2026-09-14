import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# ==========================
# 自架 CNN 模型
# ==========================

class HerbCNN(nn.Module):

    def __init__(self, num_classes=5):
        super(HerbCNN, self).__init__()


        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 2
            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 3
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(2),


            # Block 4
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()

        )


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256,512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256,num_classes)

        )



    def forward(self,x):

        x=self.features(x)

        x=self.avgpool(x)

        x=self.classifier(x)

        return x




# ==========================
# 測試圖片前處理
# ==========================

test_transform = transforms.Compose([

    transforms.Resize((224,224)),

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



    device=torch.device(

        "cuda" if torch.cuda.is_available()

        else "cpu"

    )



    # ==========================
    # 載入自架模型
    # ==========================

    model = HerbCNN(num_classes=5)


    model.load_state_dict(

        torch.load(
            "models/best_HerbCNN.pth",
            map_location=device
        )

    )


    model=model.to(device)



    print("模型載入完成 開始測試")



    model.eval()



    all_labels=[]

    all_predictions=[]



    with torch.no_grad():


        for images,labels in test_loader:


            images=images.to(device)

            labels=labels.to(device)



            outputs=model(images)



            _,predicted=torch.max(
                outputs,
                1
            )



            all_labels.extend(

                labels.cpu().numpy()

            )


            all_predictions.extend(

                predicted.cpu().numpy()

            )



    cm=confusion_matrix(

        all_labels,

        all_predictions

    )


    print(cm)



    accuracy=(

        cm.diagonal().sum()

        /

        cm.sum()

    )



    print(

        f"Accuracy: {accuracy*100:.2f}%"

    )



    class_names=[

        "Aloe",

        "Zingiber",

        "Mentha",

        "Perilla",

        "Turmeric"

    ]



    plt.figure(figsize=(8,6))



    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        xticklabels=class_names,

        yticklabels=class_names

    )



    plt.xlabel("Predicted")

    plt.ylabel("True")


    plt.savefig(

        "HerbCNN_confusion_matrix.png",

        dpi=300

    )


    plt.show()





if __name__=="__main__":

    main()
