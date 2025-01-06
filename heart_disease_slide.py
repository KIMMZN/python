import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# 데이터 로드
@st.cache_data
def load_data():
    df = pd.read_csv('./dataset/heart_disease.csv')
    return df

# 데이터 준비
df = load_data()

# 결측값 처리
if df.isnull().sum().sum() > 0:
    st.warning("데이터에 결측값이 있습니다. 결측값을 처리합니다.")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # 숫자형 데이터의 결측값은 평균으로 채움
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # 범주형 데이터의 결측값은 최빈값(모드)으로 채움
    df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

st.write("결측값 처리 완료:")
st.dataframe(df.isnull().sum())

# 주요 Feature 선택
selected_features = ['Age', 'Cholesterol Level', 'Blood Pressure']
X = df[selected_features]
y = df['Heart Disease Status']

# 데이터 분할 및 불균형 처리
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
st.write("Resampled 데이터 분포:", Counter(y_resampled))

# 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# Streamlit UI 구성
st.title("심장병 예측 시스템")
st.write("건강 정보를 입력하여 심장병 가능성을 예측합니다.")

# 사용자 입력
age = st.slider("나이 (Age)", 20, 100, 50, step=1)
cholesterol = st.number_input("콜레스테롤 수치 (Cholesterol Level)", min_value=100, max_value=400, value=200, step=1)
blood_pressure = st.number_input("혈압 (Blood Pressure)", min_value=80, max_value=200, value=120, step=1)

# 입력값 디버깅
st.write(f"입력된 나이 값: {age}")
st.write(f"입력된 콜레스테롤 값: {cholesterol}")
st.write(f"입력된 혈압 값: {blood_pressure}")

# 입력 데이터를 모델에 맞게 변환
user_data = np.array([[age, cholesterol, blood_pressure]])
user_data_scaled = scaler.transform(user_data)

# 스케일링 값 확인
st.write(f"모델에 전달되는 데이터: {user_data}")
st.write(f"스케일링된 데이터: {user_data_scaled}")

# 예측 버튼
if st.button("심장병 여부 예측"):
    prediction = model.predict(user_data_scaled)[0]
    prediction_proba = model.predict_proba(user_data_scaled)[0]

    if prediction == 1:
        st.error(f"심장병 가능성이 높습니다! (확률: {prediction_proba[1] * 100:.2f}%)")
    else:
        st.success(f"심장병 가능성이 낮습니다! (확률: {prediction_proba[0] * 100:.2f}%)")

# 특성 중요도 시각화
feature_importances = model.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
st.write("모델이 중요하게 생각하는 변수:")
st.dataframe(importance_df)
