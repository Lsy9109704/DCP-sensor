import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 页面设置
st.set_page_config(page_title="DCP/HCl Sensor AI", page_icon="🧪")
st.title("🧪 Intelligent Fluorescent Sensor Array")
st.markdown("### Rapid Identification of DCP and HCl")

# ==========================================
# 1. 加载模型 (.pkl)
# ==========================================
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            package = pickle.load(f)
        return package
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Please upload 'model.pkl' to GitHub.")
        return None

package = load_model()

if package is not None:
    # 从包裹里取出所有工具
    clf = package['classifier']
    regressors = package['regressors']
    scaler = package['scaler']
    df_sample = package['df_sample']

    # ==========================================
    # 2. 侧边栏：用户输入
    # ==========================================
    st.sidebar.header("Input Sensor Response")
    
    # 自动设置输入范围 (基于训练数据)
    min_mpa, max_mpa = df_sample['MPA'].min(), df_sample['MPA'].max()
    min_flu, max_flu = df_sample['Flu'].min(), df_sample['Flu'].max()
    mean_mpa = df_sample['MPA'].mean()
    mean_flu = df_sample['Flu'].mean()

    mpa_input = st.sidebar.number_input("Sensor 1 (MPA)", 
                                        min_value=float(min_mpa*2), 
                                        max_value=float(max_mpa*2), 
                                        value=float(mean_mpa), format="%.5f")
                                        
    flu_input = st.sidebar.number_input("Sensor 2 (Flu)", 
                                        min_value=float(min_flu*2), 
                                        max_value=float(max_flu*2), 
                                        value=float(mean_flu), format="%.5f")

    # ==========================================
    # 3. 预测逻辑
    # ==========================================
    if st.button("🔍 Analyze Unknown Gas"):
        # 预处理
        input_data = np.array([[mpa_input, flu_input]])
        input_scaled = scaler.transform(input_data)
        
        # 预测种类
        pred_species = clf.predict(input_scaled)[0]
        probs = clf.predict_proba(input_scaled)[0]
        confidence = np.max(probs) * 100
        
        # 预测浓度
        if pred_species in regressors:
            pred_log_conc = regressors[pred_species].predict(input_scaled)[0]
            pred_conc = 10 ** pred_log_conc
        else:
            pred_conc = 0.0
        
        # 显示结果
        st.success("Analysis Complete!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Identified Species", pred_species)
            st.caption(f"Confidence: {confidence:.1f}%")
        with col2:
            st.metric("Estimated Concentration", f"{pred_conc:.2f} ppb")
            st.caption("Log-scale prediction")

else:
    st.info("Please upload model.pkl to your repository.")