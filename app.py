import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==========================================
# 0. 页面配置与美化
# ==========================================
st.set_page_config(page_title="Intelligent Sensor System", page_icon="⚗️", layout="wide")

# 自定义 CSS：强制背景为白色，优化字体
st.markdown("""
<style>
    .reportview-container {
        background: #FFFFFF;
    }
    .main {
        background-color: #FFFFFF;
        color: #000000;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px; 
        font-weight: bold;
    }
    h1, h2, h3 {
        font-family: 'Arial', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚗️ Intelligent Fluorescent Sensor Array System")
st.markdown("""
**Workflow:** 1. **Species Identification** (SVM Classification based on response patterns)
2. **Concentration Prediction** (Specific Regression Model)
""")

# ==========================================
# 1. 核心逻辑：数据加载与模型训练
# ==========================================
@st.cache_resource
def build_pipeline():
    # ------------------------------------------------
    # A. 内置数据集 (DCP + HCl + Water)
    # ------------------------------------------------
    data = {
        "MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "Flu": [0.0]*18 + [-0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641] + [-0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP"]*36 + ["HCl"]*36, 
        "Concentration": [0.1]*6 + [1.0]*6 + [10.0]*6 + [100.0]*6 + [1000.0]*6 + [10000.0]*6 + [4.0]*6 + [40.0]*6 + [400.0]*6 + [4000.0]*6 + [40000.0]*6 + [400000.0]*6
    }
    
    # 加入 Water 空白数据 (防止0值误报)
    data["MPA"].extend([-0.0006, -0.0005, -0.0007, -0.0006, -0.0006, -0.0005])
    data["Flu"].extend([0.0, 0.0, 0.0, 0.0, 0.00001, -0.00001])
    data["Species"].extend(["Water"] * 6)
    data["Concentration"].extend([0] * 6)

    df = pd.DataFrame(data)

    # ------------------------------------------------
    # B. 特征工程 (取绝对值)
    # ------------------------------------------------
    # 逻辑：猝灭信号为负，取绝对值代表“响应强度”
    df['MPA_Abs'] = df['MPA'].abs()
    df['Flu_Abs'] = df['Flu'].abs()

    X = df[['MPA_Abs', 'Flu_Abs']].values
    y_spec = df['Species'].values
    y_conc = df['Concentration'].values

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------
    # C. 训练模型 (SVM Classification + SVR Regression)
    # ------------------------------------------------
    
    # 1. SVM 分类器 (Species Identification)
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_scaled, y_spec)

    # 2. SVR 回归器 (Category Information Extraction)
    # 为每种物质训练一个专属的回归模型
    regressors = {}
    for spec in ['DCP', 'HCl']:
        idx = (y_spec == spec)
        if idx.sum() > 0:
            # 使用 Log10(浓度) 进行拟合，保证线性关系
            reg = SVR(kernel='linear', C=100)
            reg.fit(X_scaled[idx], np.log10(y_conc[idx]))
            regressors[spec] = reg
            
    return clf, regressors, scaler, df

# 初始化系统
clf, regressors, scaler, df = build_pipeline()

# ==========================================
# 2. 用户交互界面 (Sidebar)
# ==========================================
with st.sidebar:
    st.header("🎛️ Sensor Response Input")
    st.info("Input the raw relative fluorescence change ($F/F_0 - 1$).")
    
    mpa_val = st.number_input("Sensor 1 (MPA)", value=-0.00250, format="%.5f", step=0.0001)
    flu_val = st.number_input("Sensor 2 (Flu)", value=0.00000, format="%.5f", step=0.0001)
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Run Analysis", type="primary")
    
    st.markdown("---")
    st.markdown("**Note:**\n- Negative value = Quenching (Target)\n- Positive value = Enhancement (Interference)")

# ==========================================
# 3. 分析逻辑与结果展示
# ==========================================
if analyze_btn:
    st.markdown("### 📊 Analysis Results")
    
    # --- 逻辑 1: 检查是否为正值 (Enhancement) ---
    # 如果两个传感器都显著大于0 (设置一个小阈值0.001防止噪音)，则为干扰物
    if mpa_val > 0.001 or flu_val > 0.001:
        st.warning("⚠️ **Unknown Interference Detected**")
        st.error("Fluorescence Enhancement effect observed. Target analytes (DCP/HCl) cause quenching (negative signal).")
        
    else:
        # --- 逻辑 2: 正常猝灭信号分析 ---
        
        # 预处理：取绝对值 -> 标准化
        input_abs = np.array([[abs(mpa_val), abs(flu_val)]])
        input_scaled = scaler.transform(input_abs)
        
        # Step 1: SVM 分类
        pred_species = clf.predict(input_scaled)[0]
        probs = clf.predict_proba(input_scaled)[0]
        confidence = np.max(probs) * 100
        
        # 结果分支
        if pred_species == "Water":
            st.success("✅ **Status: Clean (Water/Background)**")
            st.metric("Detected", "No Target", delta="Safe")
            
        else:
            # Step 2: SVR 回归预测浓度
            pred_log_conc = regressors[pred_species].predict(input_scaled)[0]
            pred_conc = 10 ** pred_log_conc # 还原 Log
            
            # 显示文本结果
            c1, c2 = st.columns(2)
            with c1:
                st.success(f"**Identified Species:** {pred_species}")
                st.caption(f"SVM Confidence: {confidence:.2f}%")
            with c2:
                st.info(f"**Predicted Conc:** {pred_conc:.2f} ppb")
                st.caption("Calculated via Linear SVR")
            
            # ==========================================
            # 4. 可视化：线性回归曲线 + 样品定位
            # ==========================================
            st.markdown("---")
            st.subheader("📈 Calibration Curve & Sample Position")
            
            # 准备绘图数据
            subset = df[df['Species'] == pred_species]
            x_train = np.log10(subset['Concentration'])
            
            # 智能选择显示哪个传感器的信号 (选择响应最强的那个)
            if pred_species == 'DCP':
                y_train = subset['MPA_Abs']
                y_sample = abs(mpa_val)
                signal_label = "|Sensor 1 (MPA)|"
                color_code = '#1f77b4' # Blue
            else: # HCl
                y_train = subset['Flu_Abs']
                y_sample = abs(flu_val)
                signal_label = "|Sensor 2 (Flu)|"
                color_code = '#2ca02c' # Green

            # 计算 R2 用于展示
            z = np.polyfit(x_train, y_train, 1) # 线性拟合
            p = np.poly1d(z)
            y_fit = p(x_train)
            r2 = r2_score(y_train, y_fit)

            # Matplotlib 绘图
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 1. 绘制标准曲线 (拟合线)
            x_line = np.linspace(x_train.min()-0.5, x_train.max()+0.5, 100)
            ax.plot(x_line, p(x_line), color=color_code, linestyle='-', linewidth=2, 
                    label=f'Linear Fit ($R^2={r2:.3f}$)')
            
            # 2. 绘制训练数据点
            ax.scatter(x_train, y_train, color='gray', alpha=0.6, s=60, label='Training Samples')
            
            # 3. 绘制用户当前样品 (红星)
            ax.scatter([pred_log_conc], [y_sample], color='red', marker='*', s=300, 
                       label='Current Sample', zorder=10, edgecolors='black')

            # 图表装饰 (符合学术标准)
            ax.set_title(f"{pred_species} Quantification Model", fontsize=14, fontweight='bold')
            ax.set_xlabel("Log10 (Concentration) [ppb]", fontsize=12)
            ax.set_ylabel(f"Signal Intensity {signal_label}", fontsize=12)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # 强制白色背景
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            
            # 在图上显示公式
            eqn = f"y = {z[0]:.4f}x + {z[1]:.4f}"
            ax.text(0.05, 0.9, eqn, transform=ax.transAxes, fontsize=11, 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            st.pyplot(fig)

else:
    st.info("👈 Please input sensor data in the sidebar to start.")
