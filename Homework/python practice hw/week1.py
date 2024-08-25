import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Maxs  = list()
Mins  = list()
Means = list()
Stds  = list()

# 設置隨機種子以確保結果可重現
np.random.seed(0)

# 生成10張隨機的三通道影像
for i in range(10):
    # print(f'Index {i}')

    images = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
    # 轉換為灰階並計算統計數據
    gray_images = np.mean(images, axis=2).astype(np.uint8)

    Maxs.append(np.max(gray_images, axis=(0, 1)))
    Mins.append(np.min(gray_images, axis=(0, 1)))
    Means.append(np.mean(gray_images, axis=(0, 1)))
    Stds.append(np.std(gray_images, axis=(0, 1)))



d = {'Max' : Maxs,
     'Min' : Mins,
     'Mean': Means,
     'Std' : Stds,
    }

df = pd.DataFrame(data=d)
df.to_excel(f'image.xlsx')