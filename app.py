import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="Smart Sensor AI", page_icon="🧪", layout="wide")

# 自定义 CSS 让背景更白，图表更清晰
st.markdown("""
<style>
    .main {background-color: #FFFFFF;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Intelligent Fluorescent Sensor Array")
st.markdown("### Rapid Identification & Concentration Analysis")
st.caption("Auto-Calibration with Integrated Dataset (DCP / HCl / Water)")

# ==========================================
# 1. 核心逻辑：内置增强版数据 (含 Water)
# ==========================================
@st.cache_resource
def train_model():
    # A. 原始数据 (DCP + HCl)
    data = {
        "MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "Flu": [0.0]*18 + [-0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641] + [-0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP"]*36 + ["HCl"]*36, 
        "Concentration": [0.1]*6 + [1.0]*6 + [10.0]*6 + [100.0]*6 + [1000.0]*6 + [10000.0]*6 + [4.0]*6 + [40.0]*6 + [400.0]*6 + [4000.0]*6 + [40000.0]*6 + [400000.0]*6
    }
    
    # B. 🔥 加入 "Water" (空白对照) 数据 🔥
    # 这是解决 0,0 误报的关键！让模型学会什么是“无”
    water_mpa = [-0.0006, -0.0005, -0.0007, -0.0006, -0.0006, -0.0005] # 基于您之前数据的基线噪音
    water_flu = [0.0, 0.0, 0.0, 0.0, 0.00001, -0.00001]
    
    data["MPA"].extend(water_mpa)
    data["Flu"].extend(water_flu)
    data["Species"].extend(["Water"] * 6)
    data["Concentration"].extend([0] * 6) # 浓度占位

    df = pd.DataFrame(data)

    # C. 训练模型
    X = df[['MPA', 'Flu']].values
    y_spec = df['Species'].values
    y_conc = df['Concentration'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. 分类器 (DCP vs HCl vs Water)
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_scaled, y_spec)

    # 2. 回归器 (只针对 DCP 和 HCl 训练)
    regressors = {}
    for spec in ['DCP', 'HCl']:
        idx = (y_spec == spec)
        # 训练 Log 浓度回归
        reg = SVR(kernel='linear', C=100)
        reg.fit(X_scaled[idx], np.log10(y_conc[idx]))
        regressors[spec] = reg
            
    return clf, regressors, scaler, df

clf, regressors, scaler, df = train_model()

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("🎛️ Sensor Input")
    st.markdown("Input relative fluorescence change ($F/F_0 - 1$).")
    
    mpa_val = st.number_input("Sensor 1 (MPA)", value=0.00000, format="%.5f", step=0.0001)
    flu_val = st.number_input("Sensor 2 (Flu)", value=0.00000, format="%.5f", step=0.0001)
    
    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Sample", type="primary")

# ==========================================
# 3. 主界面分析
# ==========================================
if analyze_btn:
    # 预处理输入
    input_data = np.array([[mpa_val, flu_val]])
    input_scaled = scaler.transform(input_data)
    
    # 预测种类
    pred_species = clf.predict(input_scaled)[0]
    confidence = np.max(clf.predict_proba(input_scaled)) * 100
    
    # --- 场景 A: 识别为水/空白 ---
    if pred_species == "Water":
        st.info("### ✅ No Target Detected")
        st.markdown("The sensor response is indistinguishable from the background (Water).")
        st.metric("Status", "Safe / Clean", delta="OK")
        
    # --- 场景 B: 识别为目标物 ---
    else:
        # 预测浓度
        pred_log = regressors[pred_species].predict(input_scaled)[0]
        pred_conc = 10 ** pred_log
        
        st.success(f"### 🚨 Detected: {pred_species}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Identified Species", pred_species)
        c2.metric("Concentration", f"{pred_conc:.2f} ppb")
        c3.metric("Confidence", f"{confidence:.1f}%")

        # ==========================================
        # 4. 可视化：线性回归曲线 + 样品定位
        # ==========================================
        st.markdown("---")
        st.subheader("📈 Calibration Curve & Sample Position")
        
        # 准备绘图数据
        subset = df[df['Species'] == pred_species]
        x_train = np.log10(subset['Concentration'])
        
        # 选择最佳响应信号画图 (DCP用MPA, HCl用Flu，这样线性度最好)
        # 或者为了统一，我们展示 X轴=Log浓度, Y轴=预测所用的主要特征
        # 这里为了直观，我们绘制 "Log(Conc) vs Principal Signal"
        if pred_species == 'DCP':
            y_train_plot = subset['MPA']
            y_sample_plot = mpa_val
            signal_name = "Sensor 1 (MPA) Signal"
            color = 'blue'
        else: # HCl
            y_train_plot = subset['Flu'] # HCl对Flu响应更明显
            y_sample_plot = flu_val
            signal_name = "Sensor 2 (Flu) Signal"
            color = 'green'

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 1. 画训练点
        ax.scatter(x_train, y_train_plot, color='gray', alpha=0.5, label='Training Data')
        
        # 2. 画拟合线
        # 为了画出平滑的线，我们预测一下范围两端
        line_x = np.linspace(x_train.min(), x_train.max(), 100)
        # 我们利用回归器的系数来反推信号，或者简单地拟合一条线用于展示
        # 这里简单拟合一条线用于视觉展示
        z = np.polyfit(x_train, y_train_plot, 1)
        p = np.poly1d(z)
        ax.plot(line_x, p(line_x), color=color, linestyle='--', linewidth=2, label='Linear Fit')
        
        # 3. 🔥 画当前样品位置 (红星) 🔥
        ax.scatter([pred_log], [y_sample_plot], color='red', s=200, marker='*', label='Current Sample', zorder=5)
        
        # 装饰
        ax.set_xlabel("Log10(Concentration) [ppb]", fontsize=10, fontweight='bold')
        ax.set_ylabel(signal_name, fontsize=10, fontweight='bold')
        ax.set_title(f"Quantitative Analysis for {pred_species}", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 纯白背景设置
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        st.pyplot(fig)

else:
    st.info("👈 Please enter sensor values and click 'Analyze'.")
