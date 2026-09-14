import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageTk
from torchvision import datasets, models, transforms

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk  # 引入現代化主題元件

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# =========================
# 中文字型設定
# =========================
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

dataset = datasets.ImageFolder(
    root="picture/train"
)

classes_en = dataset.classes
classes_cn = ["蘆薈","薄荷","紫蘇","薑黃","薑"]

# =========================
# 載入模型
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(1280, 5)

# 請確保模型路徑正確
model_path = "models/best_mobilenet_v2.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"成功載入最佳模型權重: {model_path}")
else:
    print(f"找不到 {model_path}，將使用未訓練的權重。")

model = model.to(device)
model.eval()

# 配合訓練集，改成標準 224x224 尺寸
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# =========================
# 建立視窗與現代化美化
# =========================
window = tk.Tk()
window.title("中藥材影像辨識系統")
window.geometry("900x550")  # 改為橫向寬螢幕，視覺比例更好
window.configure(bg="#F5F5F7")  # 舒服的淺灰色背景

# 設定 ttk 現代化樣式
style = ttk.Style()
style.theme_use('clam')
style.configure('TButton', font=('Microsoft JhengHei', 12), padding=6)
style.configure('Main.TFrame', background="#F5F5F7")
style.configure('Card.TFrame', background="#FFFFFF", relief="solid", borderwidth=1)

# 主容器：分為左右兩欄
main_frame = ttk.Frame(window, style='Main.TFrame', padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

left_frame = ttk.Frame(main_frame, style='Main.TFrame')
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

right_frame = ttk.Frame(main_frame, style='Main.TFrame')
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

# -------------------------
# 左側：控制與圖片區 (用白色卡片樣式包裹)
# -------------------------
image_card = ttk.Frame(left_frame, style='Card.TFrame', padding=15)
image_card.pack(fill=tk.BOTH, expand=True)

# 標題
lbl_left_title = tk.Label(image_card, text="📸 待測中藥材圖片", font=("Microsoft JhengHei", 14, "bold"), bg="#FFFFFF",
                          fg="#333333")
lbl_left_title.pack(anchor="w", pady=(0, 10))

# 圖片顯示
img_label = tk.Label(image_card, bg="#EAEDEF", text="請先選擇圖片", font=("Microsoft JhengHei", 12), width=35,
                     height=15, relief="solid", bd=1)
img_label.pack(fill=tk.BOTH, expand=True, pady=10)

# -------------------------
# 右側：結果與圖表區
# -------------------------
# 上方結果卡片
result_card = ttk.Frame(right_frame, style='Card.TFrame', padding=15)
result_card.pack(fill=tk.X, pady=(0, 15))

result_title = tk.Label(result_card, text="📊 辨識結果", font=("Microsoft JhengHei", 14, "bold"), bg="#FFFFFF",
                        fg="#333333")
result_title.pack(anchor="w")

result_label = tk.Label(
    result_card,
    text="尚未進行預測",
    font=("Microsoft JhengHei", 16),
    bg="#FFFFFF",
    fg="#0066CC",
    justify=tk.LEFT
)
result_label.pack(anchor="w", pady=(10, 0))

# 下方圖表卡片
chart_card = ttk.Frame(right_frame, style='Card.TFrame', padding=15)
chart_card.pack(fill=tk.BOTH, expand=True)

# 建立乾淨的 Matplotlib 圖表
fig = Figure(figsize=(5, 3), dpi=100, facecolor='#FFFFFF')
ax = fig.add_subplot(111)
ax.set_facecolor('#F8F9FA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CCCCCC')
ax.spines['bottom'].set_color('#CCCCCC')

canvas = FigureCanvasTkAgg(fig, master=chart_card)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


# =========================
# 預測動作函式
# =========================
def predict():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.png *.jpeg")]
    )
    if not path:
        return

    # 開啟與縮放顯示圖片
    img = Image.open(path).convert("RGB")
    show_img = img.copy()
    show_img.thumbnail((300, 300))  # 縮放到適合左側卡片的大小
    photo = ImageTk.PhotoImage(show_img)

    img_label.configure(image=photo, text="")
    img_label.image = photo

    # 模型推理
    input_img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_img)
        probability = torch.softmax(output, dim=1)

    probs = probability[0].cpu().numpy()
    confidence, index = torch.max(probability, 1)
    index = index.item()
    confidence = confidence.item() * 100

    # 動態更新結果文字
    result_label.config(
        text=f" 預測品種：{classes_cn[index]} ({classes_en[index]})\n"
             f" 辨識信心度：{confidence:.2f}%",
        fg="#2E7D32" if confidence > 70 else "#D32F2F"  # 信心度高顯示綠色，低顯示紅色
    )

    # 漂亮地更新直方圖
    ax.clear()
    bars = ax.bar(classes_cn, probs, color='#4A90E2', width=0.5, edgecolor='#357ABD', lw=1)

    # 把最高的那一條高亮塗成綠色
    bars[index].set_color('#2E7D32')
    bars[index].set_edgecolor('#1B5E20')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("機率 (Probability)", fontsize=10)
    ax.set_title("各中藥材分類機率分佈", fontsize=11, fontweight='bold', pad=10)

    # 加上百分比標籤
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height * 100:.1f}%",
            ha="center",
            fontsize=9,
            fontweight='bold' if height == probs[index] else 'normal'
        )

    fig.tight_layout()
    canvas.draw_idle()


# =========================
# 按鈕放最下面（靠左欄底）
# =========================
btn_select = ttk.Button(left_frame, text="📁 選擇藥材圖片", command=predict)
btn_select.pack(fill=tk.X, pady=(15, 0))

window.mainloop()
