import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 讀取資料
df = pd.read_excel('processed_data.xlsx')

# 類別與數值型欄位
categorical_features = ['性別', '檢傷級數', '意識程度E', '意識程度V', '意識程度M']
numerical_features = [
    '年齡', '心跳', '呼吸次數', '血壓(SBP)', '血壓(DBP)', '血氧濃度(%)',
    'pH(vein)', 'pH', 'pCO2', 'HCO3act', 'O2SAT',
    'Lactate\xa0(mmol/L)', 'D-Dimer test', 'Troponin I\xa0(ng/mL)',
    'NT-proBNP\xa0(pg/mL)', 'eGFR\xa0(mL/min/1.73 m^2)',
    'Na\xa0(mmol/L)', 'K\xa0(mmol/L)', 'BUN\xa0(mg/dL)', 'CREA\xa0(mg/dL)'
]

features = categorical_features + numerical_features
df_model = df[features + ['Y']].copy()

# 類別型特徵處理
for feature in categorical_features:
    if feature == '意識程度E':
        df_model.loc[~df_model[feature].isin([1, 2, 3, 4]), feature] = None
    elif feature == '意識程度V':
        df_model.loc[~df_model[feature].isin([1, 2, 3, 4, 5]), feature] = None
    elif feature == '意識程度M':
        df_model.loc[~df_model[feature].isin([1, 2, 3, 4, 5, 6]), feature] = None
    elif feature == '檢傷級數':
        df_model.loc[~df_model[feature].isin([1, 2, 3, 4, 5]), feature] = None
    elif feature == '性別':
        df_model.loc[~df_model[feature].isin([0, 1]), feature] = None

# 數值型轉換
for feature in numerical_features:
    df_model[feature] = pd.to_numeric(df_model[feature], errors='coerce')

df_model.dropna(inplace=True)

# 虛擬變數處理
df_dummies = pd.get_dummies(df_model, columns=categorical_features, drop_first=True)

# 拆成 X, y
X = df_dummies.drop(columns='Y')
y = df_dummies['Y']

# 分割資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 線性回歸模型
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred).astype(int)  # 四捨五入變成分類

print("R² score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("\n分類報告：")
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rounded))
