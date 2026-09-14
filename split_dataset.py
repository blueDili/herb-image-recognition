import os
from sklearn.model_selection import train_test_split
import shutil

base_dir = "picture" # Dataset root relative to this repository

classes = ["Aloe","Zingiber","Mentha","Perilla","Turmeric"] #植物類型

train_data = {}
test_data = {}

#訓練集與測試集拆分
for c in classes:
    path = os.path.join(base_dir,c)
    files = os.listdir(path)

    train, test = train_test_split(files, test_size=0.2, random_state=1)#拆分比8:2

    train_data[c] = train
    test_data[c] = test


train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")


for c in classes:
    os.makedirs(os.path.join(train_dir, c), exist_ok=True)
    os.makedirs(os.path.join(test_dir, c), exist_ok=True)

print("資料夾建立完成！")

for c in classes:

    src_dir = os.path.join(base_dir, c)

    for file in train_data[c]:
        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(train_dir, c, file)
        )

    for file in test_data[c]:
        shutil.copy(
            os.path.join(src_dir, file),
            os.path.join(test_dir, c, file)
        )

print("圖片分類完成！")

for c in classes:

    print(
        c,
        "train:",
        len(os.listdir(os.path.join(train_dir,c))),
        "test:",
        len(os.listdir(os.path.join(test_dir,c)))
    )
