import pandas as pd
import numpy as np


class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.X = None
        self.y = None

    def load_process(self):
        # 1. 智能读取 (支持 Excel 和 CSV)
        if self.filepath.endswith('.xlsx') or self.filepath.endswith('.xls'):
            print(f"正在读取 Excel 文件: {self.filepath}")
            self.df = pd.read_excel(self.filepath)
        else:
            print(f"正在读取 CSV 文件: {self.filepath}")
            try:
                self.df = pd.read_csv(self.filepath, encoding='utf-8')
            except UnicodeDecodeError:
                self.df = pd.read_csv(self.filepath, encoding='gbk')

        # 2. 提取标签 (y)
        if 'group' not in self.df.columns:
            raise ValueError("错误：找不到 'group' 列！")
        self.y = self.df['group']

        # 3. === 核心修改：只保留 'PR间期' 及其后面的所有特征 ===
        # 定位 'PR间期' 的位置
        start_col = 'PR间期'

        if start_col not in self.df.columns:
            # 容错处理：万一叫英文 'PR interval'
            if 'PR interval' in self.df.columns:
                start_col = 'PR interval'
            else:
                raise ValueError(f"错误：在表格中找不到 '{start_col}' 这一列作为起点。")

        # 获取起点列的索引
        start_idx = self.df.columns.get_loc(start_col)

        # 切片：只取从 start_idx 开始到最后一列的数据
        print(f"ℹ️ 正在截取特征：仅保留从 [{start_col}] 开始及之后的列...")
        self.X = self.df.iloc[:, start_idx:]

        # 4. 数据清洗 (针对保留下来的这些特征)
        # 仅保留数值类型 (防止混入奇怪的文本列)
        self.X = self.X.select_dtypes(include=[np.number])

        # 删除全空的列 (如果有)
        self.X = self.X.dropna(axis=1, how='all')

        # 删除缺失值严重的列 (>30%)
        threshold = 0.3 * len(self.X)
        cols_before = self.X.shape[1]
        self.X = self.X.loc[:, self.X.isnull().sum() < threshold]
        cols_after = self.X.shape[1]

        if cols_before - cols_after > 0:
            print(f"已剔除 {cols_before - cols_after} 个缺失严重的特征")

        print(f"数据加载完毕: 样本数 {self.X.shape[0]}, 特征数 {self.X.shape[1]}")
        print(f"使用的特征列表首尾: {self.X.columns[0]} ... {self.X.columns[-1]}")

        return self.X, self.y