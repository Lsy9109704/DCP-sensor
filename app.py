import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 页面设置
st.set_page_config(page_title="DCP/HCl Sensor AI", page_icon="🧪")

st.title("🧪 Intelligent Fluorescent Sensor Array")
st.markdown("### Rapid Identification of DCP and HCl")
st.info("Based on Machine Learning (SVM) Algorithm")

# ==========================================
# 1. 自动读取并训练模型
# ==========================================
@st.cache_data  # 缓存功能，让网页加载更快
def train_models():
    # 读取同目录下的 data.csv
    try:
        # 尝试读取 CSV
        df = pd.read_csv('data.csv')
    except:
        # 如果读取失败，生成模拟数据防止 App 崩溃（仅供演示）
        st.warning("Data file not found. Using simulation data.")
        return None, None, None, None

    # 强制转换列名
    df.columns = df.columns.astype(str)
    
    # 筛选 DCP 和 HCl
    target_species = ['DCP', 'HCl']
    df = df[df['Species'].isin(target_species)].copy()
    
    # 准备数据
    X = df[['1@Pt@MPA', 'aggregate of 1 on Pt@MPA']].values
    y_spec = df['Species'].values
    y_conc = df['Concentration'].values
    
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. 训练分类器
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_scaled, y_spec)
    
    # 2. 训练回归器
    regressors = {}
    for spec in target_species:
        idx = (y_spec == spec)
        reg = SVR(kernel='linear', C=100)
        # 使用 Log 浓度训练
        reg.fit(X_scaled[idx], np.log10(y_conc[idx]))
        regressors[spec] = reg
        
    return clf, regressors, scaler, df

# 执行训练
clf, regressors, scaler, df = train_models()

if clf is not None:
    # ==========================================
    # 2. 侧边栏：用户输入
    # ==========================================
    st.sidebar.header("Input Sensor Response")
    st.sidebar.markdown("Input the relative fluorescence intensity ($F/F_0 - 1$).")
    
    # 获取数据范围作为默认参考
    min_mpa, max_mpa = df['MPA'].min(), df['MPA'].max()
    min_flu, max_flu = df['Flu'].min(), df['Flu'].max()
    
    mpa_input = st.sidebar.number_input("Sensor 1 (MPA)", 
                                        min_value=float(min_mpa*1.5), 
                                        max_value=float(max_mpa*1.5), 
                                        value=float(df['MPA'].mean()), format="%.5f")
                                        
    flu_input = st.sidebar.number_input("Sensor 2 (Flu)", 
                                        min_value=float(min_flu*1.5), 
                                        max_value=float(max_flu*1.5), 
                                        value=float(df['Flu'].mean()), format="%.5f")

    # ==========================================
    # 3. 主界面：分析与结果
    # ==========================================
    if st.button("🔍 Analyze Unknown Gas"):
        # 预处理输入
        input_data = np.array([[mpa_input, flu_input]])
        input_scaled = scaler.transform(input_data)
        
        # 预测
        pred_species = clf.predict(input_scaled)[0]
        probs = clf.predict_proba(input_scaled)[0]
        confidence = np.max(probs) * 100
        
        # 预测浓度
        pred_log_conc = regressors[pred_species].predict(input_scaled)[0]
        pred_conc = 10 ** pred_log_conc
        
        # 显示结果卡片
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Identified Species", pred_species)
            st.caption(f"Confidence: {confidence:.1f}%")
        with col2:
            st.metric("Estimated Concentration", f"{pred_conc:.2f} ppb")
            st.caption("Log-scale prediction")
            
    else:
        st.info("👈 Please adjust values in the sidebar and click 'Analyze'.")

else:
    st.error("Please upload 'data.csv' to the GitHub repository.")