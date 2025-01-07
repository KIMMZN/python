#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Streamlit 제목
st.title("Heart Disease Prediction using Random Forest")

# 글자 처리 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드 및 확인
st.subheader("1. 데이터 로드 및 확인")
data = pd.read_csv('./dataset/heart_disease.csv')
st.write("데이터 샘플:")
st.dataframe(data.head())

# 2. 결측값 처리
st.subheader("2. 결측값 처리")
if data.isnull().sum().sum() > 0:
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
st.write(f"결측값 처리 완료. 남은 결측값 수: {data.isnull().sum().sum()}")

# 3. 선택된 변수 기반 데이터 분리
st.subheader("3. 변수 선택 및 데이터 분리")
selected_features = ['Age', 'Cholesterol Level', 'Blood Pressure']  # 선택된 변수
X = data[selected_features]
y = data['Heart Disease Status']

# 3.1. 데이터 불균형 해결
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

st.write("SMOTE와 데이터 분리 완료:")
st.write(f"훈련 데이터 크기: {X_train.shape}")
st.write(f"테스트 데이터 크기: {X_test.shape}")

# 4. 랜덤 포레스트 모델 학습
st.subheader("4. 랜덤 포레스트 모델 학습 및 평가")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# # 5. 모델 평가
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# st.write(f"모델 정확도: {accuracy * 100:.2f}%")
# 5. 모델 평가

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# 정확도 출력
st.write(f"모델 정확도: {accuracy * 100:.2f}%")

# 정확도 시각화 (막대그래프)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Accuracy'], [accuracy * 100], color='blue', alpha=0.7)
ax.set_ylim(0, 100)  # 0% ~ 100% 범위
ax.set_ylabel('Accuracy (%)')
ax.set_title('Model Accuracy')
for i, v in enumerate([accuracy * 100]):
    ax.text(i, v + 1, f"{v:.2f}%", ha='center', fontsize=12)
st.pyplot(fig)

# 분류 보고서
st.write("분류 보고서:")
st.text(classification_report(y_test, y_pred))

# 6. 혼동행렬 시각화
st.subheader("6. 혼동행렬 (Confusion Matrix)")

# 혼동행렬 생성
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# 혼동행렬 시각화
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted Labels")
ax.set_ylabel("True Labels")
st.pyplot(fig)

# 7. 실제값과 예측값 비교 (히스토그램)
st.subheader("7. 실제값과 예측값 비교")
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
}).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(comparison_df['Actual'], label='Actual', kde=False, bins=2, color='blue', alpha=0.6, ax=ax)
sns.histplot(comparison_df['Predicted'], label='Predicted', kde=False, bins=2, color='orange', alpha=0.6, ax=ax)
ax.set_title("Actual vs Predicted Distribution")
ax.set_xlabel("Heart Disease Status")
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# 8. 주요 변수와 타겟 변수 간 관계
st.subheader("8. 주요 변수와 심장병 간 관계")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Heart Disease Status', y='Cholesterol Level', data=data, ax=ax)
ax.set_title("Cholesterol Level by Heart Disease Status")
ax.set_xlabel("Heart Disease Status")
ax.set_ylabel("Cholesterol Level")
st.pyplot(fig)

# 9. 특성 중요도 시각화
st.subheader("9. 특성 중요도")
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
ax.set_title("Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
st.pyplot(fig)

# 10. 상관관계 히트맵
st.subheader("10. 상관관계 히트맵")

# 숫자형 데이터만 포함
correlation_data = data[selected_features + ['Heart Disease Status']].select_dtypes(include=['float64', 'int64'])

# 상관관계 계산
correlation_matrix = correlation_data.corr()

# 히트맵 그리기
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', square=True, ax=ax)
ax.set_title("Selected Features and Target Correlation Heatmap")
st.pyplot(fig)



# In[ ]:


# 사용자 입력에 따른 심장병 예측
st.subheader("8. 심장병 발병 확률 예측")

# 사용자 입력 슬라이더
age = st.slider("나이 (Age)", min_value=20, max_value=100, value=50, step=1)
cholesterol = st.slider("콜레스테롤 수치 (Cholesterol Level)", min_value=100, max_value=400, value=200, step=1)
blood_pressure = st.slider("혈압 (Blood Pressure)", min_value=80, max_value=200, value=120, step=1)

# 입력 데이터를 모델에 적용
user_data = pd.DataFrame({
    'Age': [age],
    'Cholesterol Level': [cholesterol],
    'Blood Pressure': [blood_pressure]
})

# 예측 수행
prediction = model.predict(user_data)[0]
prediction_proba = model.predict_proba(user_data)[0]

# 결과 출력
if prediction == 1:
    st.error(f"심장병 발병 가능성이 높습니다! (확률: {prediction_proba[1] * 100:.2f}%)")
else:
    st.success(f"심장병 발병 가능성이 낮습니다! (확률: {prediction_proba[0] * 100:.2f}%)")

