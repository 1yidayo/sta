import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib
import platform

df = pd.read_excel('processed_data.xlsx')

# 選擇欄位當自變數x
# 類別型
categorical_features = ['性別', '檢傷級數', '意識程度E', '意識程度V', '意識程度M'] 

#數值型
numerical_features = [
    '年齡', '心跳', '呼吸次數', '血壓(SBP)', '血壓(DBP)', '血氧濃度(%)',
    'pH(vein)', 'pH', 'pCO2', 'HCO3act', 'O2SAT',
    'Lactate\xa0(mmol/L)', 'D-Dimer test', 'Troponin I\xa0(ng/mL)',
    'NT-proBNP\xa0(pg/mL)', 'eGFR\xa0(mL/min/1.73 m^2)',
    'Na\xa0(mmol/L)', 'K\xa0(mmol/L)', 'BUN\xa0(mg/dL)', 'CREA\xa0(mg/dL)'
]

features = categorical_features + numerical_features


df_model = df[features + ['Y']].copy()

# 處理類別型特徵的異常值
for feature in categorical_features:
    if feature == '意識程度E':
        mask = ~df_model[feature].isin([1, 2, 3, 4])
        df_model.loc[mask, feature] = None
    elif feature == '意識程度V':
        mask = ~df_model[feature].isin([1, 2, 3, 4, 5])
        df_model.loc[mask, feature] = None
    elif feature == '意識程度M':
        mask = ~df_model[feature].isin([1, 2, 3, 4, 5, 6])
        df_model.loc[mask, feature] = None
    elif feature == '檢傷級數':
        mask = ~df_model[feature].isin([1, 2, 3, 4, 5])
        df_model.loc[mask, feature] = None
    elif feature == '性別':
        mask = ~df_model[feature].isin([0, 1])
        df_model.loc[mask, feature] = None

# 處理數值型特徵的異常值
for feature in numerical_features:
    df_model[feature] = pd.to_numeric(df_model[feature], errors='coerce')

df_model.dropna(inplace=True) # 丟掉有缺失值的行

# 自變數和因變數
X = df_model.drop(columns='Y')
y = df_model['Y']

# 使用Smote 合成樣本數少的 因為類別樣本數不平衡 避免模型偏向樣本數多的類別
sm = SMOTE(random_state=42)  
X_resampled, y_resampled = sm.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 建立模型
model = RandomForestClassifier(
    n_estimators=1000,      # 決策樹數量
    max_depth=None,        # 樹最大深度，可調
    random_state=42
)

model.fit(X_train, y_train) # 訓練模型

y_pred = model.predict(X_test) # 模型預測測試集

print("\n分類報告:\n", classification_report(y_test, y_pred))
print("\n混淆矩陣:\n", confusion_matrix(y_test, y_pred))

importances = model.feature_importances_
feature_names = X.columns

if platform.system() == 'Windows':
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel('Feature Importance')
plt.title('重要特徵分析')
plt.tight_layout()
plt.show()