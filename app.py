import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# ==========================================
# 0. 页面配置
# ==========================================
st.set_page_config(page_title="Intelligent Sensor System", page_icon="⚗️", layout="wide")

st.markdown("""
<style>
    .main {background-color: #FFFFFF;}
    h1, h2, h3 {font-family: 'Arial', sans-serif;}
</style>
""", unsafe_allow_html=True)

st.title("⚗️ Intelligent Fluorescent Sensor Array System")
st.markdown("""
**Workflow:** 1. **Classification (SVM):** Uses ALL sensors to identify the species.
2. **Quantification (Linear Regression):** Uses the PRIMARY sensor to calculate concentration (Standard Curve Method).
""")

# ==========================================
# 1. 核心逻辑：分别训练分类器和回归器
# ==========================================
@st.cache_resource
def build_pipeline():
    # 数据集
    data = {
        "1@Pt@MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "aggregate of 1 on Pt@MPA": [0.0]*18 + [-0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641] + [-0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP"]*36 + ["HCl"]*36, 
        "Concentration": [0.1]*6 + [1.0]*6 + [10.0]*6 + [100.0]*6 + [1000.0]*6 + [10000.0]*6 + [4.0]*6 + [40.0]*6 + [400.0]*6 + [4000.0]*6 + [40000.0]*6 + [400000.0]*6
    }
    
    # 增加 Water 对照组
    data["1@Pt@MPA"].extend([-0.0006, -0.0005, -0.0007, -0.0006, -0.0006, -0.0005])
    data["aggregate of 1 on Pt@MPA"].extend([0.0, 0.0, 0.0, 0.0, 0.00001, -0.00001])
    data["Species"].extend(["Water"] * 6)
    data["Concentration"].extend([0] * 6)

    df = pd.DataFrame(data)

    # 取绝对值
    df['1@Pt@MPA_Abs'] = df['1@Pt@MPA'].abs()
    df['aggregate of 1 on Pt@MPA_Abs'] = df['aggregate of 1 on Pt@MPA'].abs()

    # 1. 训练 SVM 分类器 (用两个传感器)
    X_cls = df[['1@Pt@MPA_Abs', 'aggregate of 1 on Pt@MPA_Abs']].values
    y_cls = df['Species'].values
    scaler = StandardScaler()
    X_cls_scaled = scaler.fit_transform(X_cls)
    
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_cls_scaled, y_cls)

    # 2. 训练 Linear Regression 回归器 (只用最佳传感器)
    regressors = {}
    
    # --- DCP 模型 (只用 MPA) ---
    dcp_df = df[df['Species'] == 'DCP']
    reg_dcp = LinearRegression()
    # X: Log(Conc), Y: MPA_Abs
    # 注意：为了预测浓度，我们通常拟合 X=Signal, Y=Log(Conc)
    reg_dcp.fit(dcp_df[['MPA_Abs']], np.log10(dcp_df['Concentration']))
    regressors['DCP'] = reg_dcp
    
    # --- HCl 模型 (只用 Flu) ---
    hcl_df = df[df['Species'] == 'HCl']
    reg_hcl = LinearRegression()
    reg_hcl.fit(hcl_df[['Flu_Abs']], np.log10(hcl_df['Concentration']))
    regressors['HCl'] = reg_hcl
            
    return clf, regressors, scaler, df

clf, regressors, scaler, df = build_pipeline()

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("🎛️ Sensor Input")
    st.info("Input raw values ($F/F_0 - 1$)")
   1@Pt@MPA_val = st.number_input("Sensor 1 (1@Pt@MPA)", value=-0.00250, format="%.5f", step=0.0001)
    aggregate of 1 on Pt@MPA_val = st.number_input("Sensor 2 (aggregate of 1 on Pt@MPA)", value=0.00000, format="%.5f", step=0.0001)
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze", type="primary")

# ==========================================
# 3. 分析逻辑
# ==========================================
if analyze_btn:
    # 检查正值干扰
    if 1@Pt@MPA_val > 0.001 or flu_val > 0.001:
        st.error("⚠️ Unknown Interference (Enhancement Detected)")
    else:
        # 1. 分类 (SVM)
        input_abs = np.array([[abs(1@Pt@MPA_val_val), abs(flu_val)]])
        input_scaled = scaler.transform(input_abs)
        pred_species = clf.predict(input_scaled)[0]
        confidence = np.max(clf.predict_proba(input_scaled)) * 100
        
        if pred_species == "Water":
            st.success("✅ Clean / Background")
        else:
            # 2. 浓度计算 (Linear Regression)
            if pred_species == 'DCP':
                # DCP 只看 MPA
                pred_log = regressors['DCP'].predict([[abs(1@Pt@MPA_val)]])[0]
                signal_used = abs(1@Pt@MPA_val)
                signal_name = "1@Pt@MPA_val (Abs)"
                color = 'blue'
            else:
                # HCl 只看 Flu
                pred_log = regressors['HCl'].predict([[abs(aggregate of 1 on Pt@MPA_val)]])[0]
                signal_used = abs(aggregate of 1 on Pt@MPA_val)
                signal_name = "aggregate of 1 on Pt@MPA (Abs)"
                color = 'green'
                
            pred_conc = 10 ** pred_log
            
            # 显示结果
            c1, c2 = st.columns(2)
            c1.metric("Species", pred_species, f"{confidence:.1f}% Conf.")
            c2.metric("Concentration", f"{pred_conc:.2f} ppb")
            
            # 3. 绘图 (完全匹配算法)
            st.markdown("---")
            st.subheader("📈 Standard Calibration Curve")
            
            subset = df[df['Species'] == pred_species]
            
            if pred_species == 'DCP':
                x_data = subset['1@Pt@MPA_Abs'] # X轴: 信号
            else:
                x_data = subset['aggregate of 1 on Pt@MPA_Abs'] # X轴: 信号
                
            y_data = np.log10(subset['Concentration']) # Y轴: 浓度
            
            # 重新拟合用于画图 (反转坐标轴以便观看习惯: X=浓度, Y=信号)
            # 注意：上面的模型是 Signal -> Conc。画图通常画 Conc -> Signal。
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # 画训练点 (X: Log Conc, Y: Signal)
            ax.scatter(y_data, x_data, color='gray', alpha=0.6, label='Standards')
            
            # 画拟合线
            z = np.polyfit(y_data, x_data, 1) # 拟合 Signal = a * Log(Conc) + b
            p = np.poly1d(z)
            line_x = np.linspace(y_data.min(), y_data.max(), 100)
            ax.plot(line_x, p(line_x), color=color, linestyle='--', label='Linear Fit')
            
            # 画当前点
            ax.scatter([pred_log], [signal_used], color='red', marker='*', s=300, label='Your Sample', zorder=10)
            
            ax.set_xlabel("Log10 (Concentration)", fontweight='bold')
            ax.set_ylabel(f"Signal Intensity {signal_name}", fontweight='bold')
            ax.set_title(f"{pred_species} Calibration Curve", fontweight='bold')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.5)
            
            # 公式
            r2 = r2_score(x_data, p(y_data))
            eq = f"Signal = {z[0]:.4f} * Log(C) + {z[1]:.4f}\n$R^2$ = {r2:.3f}"
            ax.text(0.05, 0.85, eq, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9))
            
            st.pyplot(fig)



