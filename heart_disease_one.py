#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

# Streamlit 제목
st.title("Heart Disease Analysis and Prediction using Random Forest")

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
    categorical_columns = data.select_dtypes(include=['object']).columns

    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

st.write(f"결측값 처리 완료. 남은 결측값 수: {data.isnull().sum().sum()}")

# 3. 데이터 인코딩
st.subheader("3. 범주형 데이터 인코딩")
label_encoder = LabelEncoder()
categorical_columns = data.select_dtypes(include=['object']).columns
for col in categorical_columns:
    data[col] = label_encoder.fit_transform(data[col])
st.write("인코딩 완료:")
st.dataframe(data.head())

# 4. 데이터 분리 및 불균형 해결
st.subheader("4. 데이터 분리 및 불균형 해결")
X = data.drop(columns=['Heart Disease Status'])
y = data['Heart Disease Status']

# SMOTE를 사용해 데이터 불균형 해결
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
st.write("SMOTE와 데이터 분리 완료:")
st.write(f"훈련 데이터 크기: {X_train.shape}")
st.write(f"테스트 데이터 크기: {X_test.shape}")

# 5. 랜덤 포레스트 모델 학습
st.subheader("5. 랜덤 포레스트 모델 학습")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6. 모델 평가
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"모델 정확도: {accuracy * 100:.2f}%")

#6.5 실제값과 예측값 비교
# 6.5 실제값과 예측값 비교
st.subheader("6.5 실제값과 예측값 비교")

# 데이터프레임 생성
comparison_df = pd.DataFrame({
    'Actual': y_test,         # 실제값
    'Predicted': y_pred       # 예측값
}).reset_index(drop=True)

# 1. 오차(차이) 시각화
comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Predicted']
fig1, ax1 = plt.subplots(figsize=(10, 6))
comparison_df['Error'].value_counts().plot(kind='bar', ax=ax1, color=['blue', 'orange'])
ax1.set_title("Distribution of Errors (Actual - Predicted)")
ax1.set_xlabel("Error")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# 2. 샘플링된 선 그래프
sampled_df = comparison_df.sample(n=100, random_state=42).sort_index()  # 랜덤 샘플링
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(sampled_df.index, sampled_df['Actual'], label='Actual', color='blue', alpha=0.7)
ax2.plot(sampled_df.index, sampled_df['Predicted'], label='Predicted', color='orange', alpha=0.7)
ax2.set_title("Actual vs Predicted (Sampled Data)")
ax2.set_xlabel("Index")
ax2.set_ylabel("Heart Disease Status")
ax2.legend()
st.pyplot(fig2)

# 3. 히스토그램 분포 비교
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.histplot(comparison_df['Actual'], label='Actual', kde=True, bins=2, color='blue', alpha=0.6, ax=ax3)
sns.histplot(comparison_df['Predicted'], label='Predicted', kde=True, bins=2, color='orange', alpha=0.6, ax=ax3)
ax3.set_title("Actual vs Predicted Distribution")
ax3.set_xlabel("Heart Disease Status")
ax3.set_ylabel("Count")
ax3.legend()
st.pyplot(fig3)


# 분류 보고서
report = classification_report(y_test, y_pred, target_names=['질병 없음', '질병 있음'], output_dict=True)
st.write("분류 보고서:")
st.text(classification_report(y_test, y_pred))

# 7. 상관관계 히트맵
st.subheader("7. 상관관계 히트맵")
correlation_matrix = data.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', square=True, ax=ax)
ax.set_title("변수별 상관관계 히트맵")
st.pyplot(fig)

#8. 주요 변수와 타겟 변수 간 관계
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
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), ax=ax)
ax.set_title("Feature Importances")
ax.set_xlabel("Importance")
ax.set_ylabel("Features")
st.pyplot(fig)

