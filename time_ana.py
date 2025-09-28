import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
path = "/home/gyx/projects/shapeformer/Dataset/raw/serinf2jag/modisf/20250304_0932_l17_619/train_converted_data_5D.csv"
data = pd.read_csv(path)

# 确保 'utc' 列存在并且是数值类型
if 'utc' in data.columns:
    data['utc'] = pd.to_datetime(data['utc'], errors='coerce')
    data = data.dropna(subset=['utc'])
    data['utc'] = data['utc'].astype('int64') // 10**9  # Convert to Unix timestamp in seconds

    # 绘制折线图
    plt.plot(data['utc'])
    plt.xlabel('Index')
    plt.ylabel('UTC')
    plt.title('UTC Line Plot')

    # 保存图片到本地
    plt.savefig('/home/gyx/projects/mrf/utc_line_plot.png')
else:
    print("The 'utc' column is not present in the CSV file.")
