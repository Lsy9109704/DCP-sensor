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

# 纯白背景与样式优化
st.markdown("""
<style>
    .main {background-color: #FFFFFF;}
    div[data-testid="stMetricValue"] {font-size: 24px;}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Intelligent Fluorescent Sensor Array")
st.markdown("### Rapid Identification & Concentration Analysis")
st.caption("Logic Update: Using Signal Magnitude (Absolute Value) | Higher Signal = Higher Concentration")

# ==========================================
# 1. 核心逻辑：使用绝对值训练 (解决负号直觉问题)
# ==========================================
@st.cache_resource
def train_model():
    # 原始数据
    data = {
        "MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "Flu": [0.0]*18 + [-0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641] + [-0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP"]*36 + ["HCl"]*36, 
        "Concentration": [0.1]*6 + [1.0]*6 + [10.0]*6 + [100.0]*6 + [1000.0]*6 + [10000.0]*6 + [4.0]*6 + [40.0]*6 + [400.0]*6 + [4000.0]*6 + [40000.0]*6 + [400000.0]*6
    }
    
    # 增加 Water 数据
    data["MPA"].extend([-0.0006, -0.0005, -0.0007, -0.0006, -0.0006, -0.0005])
    data["Flu"].extend([0.0, 0.0, 0.0, 0.0, 0.00001, -0.00001])
    data["Species"].extend(["Water"] * 6)
    data["Concentration"].extend([0] * 6)

    df = pd.DataFrame(data)

    # 🔥🔥🔥 关键修改：取绝对值 (Quenching Magnitude) 🔥🔥🔥
    df['MPA_Abs'] = df['MPA'].abs()
    df['Flu_Abs'] = df['Flu'].abs()

    # 使用绝对值特征进行训练
    X = df[['MPA_Abs', 'Flu_Abs']].values
    y_spec = df['Species'].values
    y_conc = df['Concentration'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. 分类器
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_scaled, y_spec)

    # 2. 回归器
    regressors = {}
    for spec in ['DCP', 'HCl']:
        idx = (y_spec == spec)
        # Log 浓度 vs 绝对值信号
        reg = SVR(kernel='linear', C=100)
        reg.fit(X_scaled[idx], np.log10(y_conc[idx]))
        regressors[spec] = reg
            
    return clf, regressors, scaler, df

clf, regressors, scaler, df = train_model()

# ==========================================
# 2. 侧边栏 (允许输入正数或负数，自动处理)
# ==========================================
with st.sidebar:
    st.header("🎛️ Sensor Input")
    st.markdown("Enter Quenching Magnitude (Absolute value preferred).")
    
    # 允许用户输入正数或负数，我们后台自动转绝对值
    mpa_val = st.number_input("Sensor 1 (MPA) Signal", value=0.00000, format="%.5f", step=0.0001)
    flu_val = st.number_input("Sensor 2 (Flu) Signal", value=0.00000, format="%.5f", step=0.0001)
    
    st.markdown("---")
    analyze_btn = st.button("🔍 Analyze Sample", type="primary")

# ==========================================
# 3. 主界面分析
# ==========================================
if analyze_btn:
    # 自动处理为绝对值 (防止用户输入负号导致逻辑错误)
    mpa_abs = abs(mpa_val)
    flu_abs = abs(flu_val)
    
    # 预处理输入
    input_data = np.array([[mpa_abs, flu_abs]])
    input_scaled = scaler.transform(input_data)
    
    # 预测种类
    pred_species = clf.predict(input_scaled)[0]
    confidence = np.max(clf.predict_proba(input_scaled)) * 100
    
    # 场景 A: 水/空白
    if pred_species == "Water":
        st.info("### ✅ No Target Detected")
        st.markdown("Signal is too weak (Background noise).")
        st.metric("Status", "Clean", delta="Safe")
        
    # 场景 B: 目标物
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
        # 4. 可视化：正斜率线性回归曲线
        # ==========================================
        st.markdown("---")
        st.subheader(f"📈 {pred_species} Quantification Curve")
        
        subset = df[df['Species'] == pred_species]
        x_train = np.log10(subset['Concentration'])
        
        # 选择最佳响应信号画图
        if pred_species == 'DCP':
            y_train_plot = subset['MPA_Abs'] # 使用绝对值
            y_sample_plot = mpa_abs
            signal_name = "|Sensor 1 (MPA) Signal|"
            color = 'royalblue'
        else: # HCl
            y_train_plot = subset['Flu_Abs'] # 使用绝对值
            y_sample_plot = flu_abs
            signal_name = "|Sensor 2 (Flu) Signal|"
            color = 'forestgreen'

        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 1. 画训练点 (现在是正相关的点)
        ax.scatter(x_train, y_train_plot, color='gray', alpha=0.5, s=50, label='Training Data')
        
        # 2. 画拟合线 (现在是正斜率 / 向上走)
        z = np.polyfit(x_train, y_train_plot, 1)
        p = np.poly1d(z)
        line_x = np.linspace(x_train.min(), x_train.max(), 100)
        ax.plot(line_x, p(line_x), color=color, linestyle='--', linewidth=2, label='Linear Regression')
        
        # 3. 画当前样品
        ax.scatter([pred_log], [y_sample_plot], color='red', s=250, marker='*', label='Your Sample', zorder=10)
        
        # 坐标轴和标签
        ax.set_xlabel("Log10(Concentration) [ppb]", fontsize=11, fontweight='bold')
        ax.set_ylabel(f"Quenching Intensity\n{signal_name}", fontsize=11, fontweight='bold')
        ax.set_title(f"Higher Signal = Higher Concentration", fontsize=12)
        
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # 纯白去边框风格
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        sns.despine()
        
        st.pyplot(fig)

else:
    st.info("👈 Please enter sensor values (absolute values) and click 'Analyze'.")
