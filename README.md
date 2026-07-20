# house-prices-prediction
Kaggle房价预测项目

## 项目说明
使用 Ames 房价数据集，通过 XGBoost 构建房价预测模型。    
这是个人机器学习入门后的第一个完整实战项目。

## 最终成绩
Kaggle Public LB: **0.12884**

## 依赖库
pandas, numpy, xgboost

## 项目结构
House_Prices_01.py   # 主程序  
submission.csv       # 预测结果  
README.md

## 核心特征
TotalSF：地下室 + 一楼 + 二楼  
Qual_GrLivArea：质量 × 地上居住面积  
Qual_Kitchen_GrLiv：厨房质量 × 地上居住面积  
Neighborhood_AvgPrice：街区平均房价  

## 备注
从数据清洗到模型调优，完整走完了一遍流程。  
分数不算高，但这是我从零到一完整走完的第一个项目。每一步都是自己调的。  
如果你也是刚入门机器学习，欢迎交流。如果发现代码有什么可以优化的地方，也请多多指教。  
