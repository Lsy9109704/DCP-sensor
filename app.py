import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="DCP/HCl Sensor AI", page_icon="🧪")
st.title("🧪 Intelligent Fluorescent Sensor Array")
st.markdown("### Rapid Identification of DCP and HCl")
st.caption("Based on integrated dataset (n=72) | Auto-Training on Startup")

# ==========================================
# 1. 核心逻辑：内置数据与现场训练
# ==========================================
@st.cache_resource
def get_model():
    # ---------------------------------------------------------
    # A. 直接内置数据 (您的真实实验数据)
    # ---------------------------------------------------------
    data_dict = {
        "MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "Flu": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641, -0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "DCP", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl", "HCl"], 
        "Concentration": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 10000.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 40000.0, 40000.0, 40000.0, 40000.0, 40000.0, 40000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0]
    }
    df = pd.DataFrame(data_dict)

    # ---------------------------------------------------------
    # B. 现场训练 (完美解决版本不兼容问题)
    # ---------------------------------------------------------
    X = df[['MPA', 'Flu']].values
    y_spec = df['Species'].values
    y_conc = df['Concentration'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. 训练分类器 (SVM)
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_scaled, y_spec)

    # 2. 训练回归器 (SVR)
    regressors = {}
    for spec in ['DCP', 'HCl']:
        idx = (y_spec == spec)
        if np.sum(idx) > 0:
            reg = SVR(kernel='linear', C=100)
            # 使用 Log10 浓度进行训练，解决跨度大问题
            reg.fit(X_scaled[idx], np.log10(y_conc[idx]))
            regressors[spec] = reg
            
    return clf, regressors, scaler, df

# 获取训练好的模型 (利用缓存，只会训练一次，速度很快)
clf, regressors, scaler, df_sample = get_model()

# ==========================================
# 2. 用户界面 (侧边栏)
# ==========================================
st.sidebar.header("Input Sensor Response")
st.sidebar.markdown("Input the fluorescence intensity values.")

# 获取数据范围作为参考
mean_mpa = df_sample['MPA'].mean()
mean_flu = df_sample['Flu'].mean()

# 输入框
mpa_input = st.sidebar.number_input("Sensor 1 (MPA)", value=float(mean_mpa), format="%.5f")
flu_input = st.sidebar.number_input("Sensor 2 (Flu)", value=float(mean_flu), format="%.5f")

st.sidebar.markdown("---")
st.sidebar.info("Click **Analyze** to predict.")

# ==========================================
# 3. 预测逻辑与结果显示
# ==========================================
if st.button("🔍 Analyze Unknown Gas"):
    # 1. 数据预处理
    input_data = np.array([[mpa_input, flu_input]])
    input_scaled = scaler.transform(input_data)
    
    # 2. 预测种类
    pred_species = clf.predict(input_scaled)[0]
    probs = clf.predict_proba(input_scaled)[0]
    confidence = np.max(probs) * 100
    
    # 3. 预测浓度
    if pred_species in regressors:
        # 预测出来的是 Log 值，需要还原
        pred_log_conc = regressors[pred_species].predict(input_scaled)[0]
        pred_conc = 10 ** pred_log_conc
    else:
        pred_conc = 0.0
    
    # 4. 漂亮的结果展示
    st.success("Analysis Complete!")
    
    # 使用两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Identified Species", pred_species)
        st.caption(f"Confidence: {confidence:.2f}%")
        
    with col2:
        st.metric("Estimated Concentration", f"{pred_conc:.2f} ppb")
        st.caption("Log-scale Regression")

    st.markdown("---")
    st.markdown(f"**Diagnosis Logic:** The system detected a pattern consistent with **{pred_species}**. The concentration was estimated using a specialized SVR model trained on {pred_species} data.")

else:
    st.info("👈 Please enter sensor values in the sidebar.")
