import numpy as np
import pandas as pd
from xgboost import XGBRegressor
import os

# ---------- 1. 切换到脚本所在目录（解决路径问题） ----------
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# ---------- 2. 数据清洗函数 ----------
# 作用：合并训练集和测试集后，统一进行缺失值填充和独热编码
# 策略：
#   - 缺失率 > 40%：直接删除该列
#   - 缺失率 10%~40%：数值列用中位数填充，文本列用众数填充
#   - 缺失率 < 10%：统一用众数填充
def auto_fill_merged(df):
    df = df.copy()

    for col in df.columns:
        # 跳过辅助列（source）和目标列（SalePrice）
        if col in ['source', 'SalePrice']:
            continue

        # 计算缺失比例
        miss = df[col].isnull().sum() / len(df[col])

        # 缺失率 > 40%：直接删除
        if miss >= 0.4:
            df.drop(columns=[col], inplace=True)
            continue

        # 缺失率 >= 10%：按类型分别填充
        if miss >= 0.1:
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_val = df[col].median()   # 数值列用中位数
            else:
                fill_val = df[col].mode()[0]  # 文本列用众数
        else:
            # 缺失率 < 10%：统一用众数
            fill_val = df[col].mode()[0]

        df[col] = df[col].fillna(fill_val)

    # 全部填充完成后，统一做独热编码
    df_clean = pd.get_dummies(df, dtype=int)
    return df_clean


# ---------- 3. 加载数据 ----------
train_data = pd.read_csv('data/train.csv', encoding='utf-8')
test_data = pd.read_csv('data/test.csv', encoding='utf-8')
test_ids = test_data['Id']   # 备份测试集ID（后续合并后会被处理掉）


# ---------- 4. 特征工程（人工构造高价值特征） ----------
# 4.1 总面积 = 地下室 + 一楼 + 二楼
train_data['TotalSF'] = (
    train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
)
test_data['TotalSF'] = (
    test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']
)

# 4.2 质量 × 总面积（捕捉“大面积+高质量”的倍增效应）
train_data['Qual_TotalSF'] = train_data['OverallQual'] * train_data['TotalSF']
test_data['Qual_TotalSF'] = test_data['OverallQual'] * test_data['TotalSF']

# 4.3 质量 × 地上居住面积（最有效的特征之一）
train_data['Qual_GrLivArea'] = train_data['OverallQual'] * train_data['GrLivArea']
test_data['Qual_GrLivArea'] = test_data['OverallQual'] * test_data['GrLivArea']

# 4.4 厨房质量 × 面积（需要先把文字等级映射成数字）
kitchen_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
train_data['KitchenQual_Num'] = train_data['KitchenQual'].map(kitchen_map).fillna(0)
test_data['KitchenQual_Num'] = test_data['KitchenQual'].map(kitchen_map).fillna(0)
train_data['Qual_Kitchen_GrLiv'] = train_data['KitchenQual_Num'] * train_data['GrLivArea']
test_data['Qual_Kitchen_GrLiv'] = test_data['KitchenQual_Num'] * test_data['GrLivArea']

# 4.5 街区平均房价（目标编码：把“街区”这个类别信息转化为数值信号）
neighborhood_avg = train_data.groupby('Neighborhood')['SalePrice'].mean()
train_data['Neighborhood_AvgPrice'] = train_data['Neighborhood'].map(neighborhood_avg)
test_data['Neighborhood_AvgPrice'] = test_data['Neighborhood'].map(neighborhood_avg)

# 测试集可能出现训练集没见过的街区，用全局平均值填充
global_avg = train_data['SalePrice'].mean()
test_data['Neighborhood_AvgPrice'] = test_data['Neighborhood_AvgPrice'].fillna(global_avg)
train_data['Neighborhood_AvgPrice'] = train_data['Neighborhood_AvgPrice'].fillna(global_avg)


# ---------- 5. 合并训练集和测试集（统一处理） ----------
train_data['source'] = 1   # 标记来源：训练集
test_data['source'] = 0    # 标记来源：测试集

all_data = pd.concat([train_data, test_data], ignore_index=True, sort=False)
all_clean = auto_fill_merged(all_data)   # 统一清洗 + 独热编码

# 拆分回训练集和测试集
train_clean = all_clean[all_clean['source'] == 1].drop(columns='source')
test_clean = all_clean[all_clean['source'] == 0].drop(columns='source')


# ---------- 6. 模型训练与预测 ----------
# 分离特征和目标（目标取对数，因为 Kaggle 用 RMSE 对数误差评估）
X_train = train_clean.drop(columns='SalePrice')
y_train_log = np.log1p(train_clean['SalePrice'])

# 测试集只保留训练集中用到的特征列（列顺序和数量必须完全一致）
X_test = test_clean[X_train.columns]

# XGBoost 模型（核心参数：500棵树，学习率0.05，深度6）
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train_log)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)   # 还原为原始房价


# ---------- 7. 生成提交文件 ----------
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': y_pred
})
submission.to_csv('submission.csv', index=False)