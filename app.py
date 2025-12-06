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
**Workflow:**
1. **Classification:** Identify Species (DCP vs HCl).
2. **Robust Quantification:** Linear Regression with **Outlier Removal** (2$\sigma$ threshold).
""")

# ==========================================
# 1. 核心逻辑：带异常值剔除的训练管道
# ==========================================
@st.cache_resource
def build_pipeline():
    # --- A. 数据集 ---
    data = {
        "MPA": [-0.00238, -0.00242, -0.00237, -0.00238, -0.0024, -0.0024, -0.00269, -0.00267, -0.00268, -0.0027, -0.00262, -0.00268, -0.00478, -0.0047, -0.00477, -0.00472, -0.00473, -0.00473, -0.00794, -0.00801, -0.00793, -0.00798, -0.00799, -0.00799, -0.01043, -0.01043, -0.01042, -0.01047, -0.01045, -0.01044, -0.02717, -0.02718, -0.02718, -0.02714, -0.0272, -0.02718, -0.00158, -0.00152, -0.00156, -0.00157, -0.0016, -0.00159, -0.0032, -0.00317, -0.00313, -0.00317, -0.00319, -0.00317, -0.00559, -0.0056, -0.00557, -0.0056, -0.00559, -0.00559, -0.00675, -0.00684, -0.00694, -0.00643, -0.00687, -0.00684, -0.01204, -0.01241, -0.01235, -0.01989, -0.0121, -0.01202, -0.02165, -0.0215, -0.0208, -0.0211, -0.02088, -0.0211], 
        "Flu": [0.0]*18 + [-0.00103, -0.00124, -0.00162, -0.00133, -0.00144, -0.0011, -0.00371, -0.00372, -0.00381, -0.00372, -0.00372, -0.00311, -0.00648, -0.00648, -0.00626, -0.00635, -0.00648, -0.00641] + [-0.00414, -0.00413, -0.00512, -0.00453, -0.00499, -0.00422, -0.00514, -0.00513, -0.00512, -0.00553, -0.00588, -0.00522, -0.00604, -0.00681, -0.00691, -0.00612, -0.00632, -0.00632, -0.01269, -0.01275, -0.01267, -0.01245, -0.01303, -0.01279, -0.0221, -0.02241, -0.02237, -0.02242, -0.02299, -0.02301, -0.04369, -0.04188, -0.04581, -0.04172, -0.04128, -0.04193], 
        "Species": ["DCP"]*36 + ["HCl"]*36, 
        "Concentration": [0.1]*6 + [1.0]*6 + [10.0]*6 + [100.0]*6 + [1000.0]*6 + [10000.0]*6 + [4.0]*6 + [40.0]*6 + [400.0]*6 + [4000.0]*6 + [40000.0]*6 + [400000.0]*6
    }
    # 补充 Water 对照
    data["MPA"].extend([-0.0006, -0.0005, -0.0007, -0.0006, -0.0006, -0.0005])
    data["Flu"].extend([0.0, 0.0, 0.0, 0.0, 0.00001, -0.00001])
    data["Species"].extend(["Water"] * 6)
    data["Concentration"].extend([0] * 6)

    df = pd.DataFrame(data)
    df['MPA_Abs'] = df['MPA'].abs()
    df['Flu_Abs'] = df['Flu'].abs()

    # --- B. 训练分类器 (SVM) ---
    scaler = StandardScaler()
    X_cls = scaler.fit_transform(df[['MPA_Abs', 'Flu_Abs']].values)
    clf = SVC(kernel='linear', C=100, probability=True)
    clf.fit(X_cls, df['Species'].values)

    # --- C. 训练回归器 (带异常剔除) ---
    models_info = {} # 存储模型和清洗后的数据
    
    for spec in ['DCP', 'HCl']:
        # 1. 提取对应数据
        subset = df[df['Species'] == spec].copy()
        if spec == 'DCP':
            y_raw = subset['MPA_Abs'].values # 信号
            X_raw = np.log10(subset['Concentration'].values).reshape(-1, 1) # Log浓度
        else:
            y_raw = subset['Flu_Abs'].values
            X_raw = np.log10(subset['Concentration'].values).reshape(-1, 1)

        # 2. 第一轮拟合
        reg_initial = LinearRegression()
        reg_initial.fit(X_raw, y_raw)
        preds = reg_initial.predict(X_raw)
        
        # 3. 计算残差并识别异常值 ( > 2倍标准差)
        residuals = np.abs(y_raw - preds)
        threshold = 2.0 * np.std(residuals)
        mask = residuals < threshold # True表示保留，False表示异常
        
        X_clean = X_raw[mask]
        y_clean = y_raw[mask]
        X_out = X_raw[~mask]
        y_out = y_raw[~mask]
        
        # 4. 第二轮拟合 (只用干净数据)
        reg_final = LinearRegression()
        reg_final.fit(X_clean, y_clean)
        
        models_info[spec] = {
            'model': reg_final,
            'X_clean': X_clean,
            'y_clean': y_clean,
            'X_out': X_out,
            'y_out': y_out,
            'feature': 'MPA_Abs' if spec == 'DCP' else 'Flu_Abs'
        }
            
    return clf, models_info, scaler, df

clf, models_info, scaler, df = build_pipeline()

# ==========================================
# 2. 侧边栏
# ==========================================
with st.sidebar:
    st.header("🎛️ Sensor Input")
    st.info("Input raw values ($F/F_0 - 1$)")
    mpa_val = st.number_input("Sensor 1 (MPA)", value=-0.00250, format="%.5f", step=0.0001)
    flu_val = st.number_input("Sensor 2 (Flu)", value=0.00000, format="%.5f", step=0.0001)
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze", type="primary")

# ==========================================
# 3. 分析逻辑
# ==========================================
if analyze_btn:
    if mpa_val > 0.001 or flu_val > 0.001:
        st.error("⚠️ Unknown Interference (Enhancement Detected)")
    else:
        # 1. 分类
        input_abs = np.array([[abs(mpa_val), abs(flu_val)]])
        pred_species = clf.predict(scaler.transform(input_abs))[0]
        
        if pred_species == "Water":
            st.success("✅ Clean / Background")
        else:
            # 2. 定量 (使用清洗后的模型)
            model_data = models_info[pred_species]
            reg = model_data['model']
            
            # 使用对应传感器信号预测
            signal_val = abs(mpa_val) if pred_species == 'DCP' else abs(flu_val)
            
            # 求解浓度: Signal = a * Log(C) + b  =>  Log(C) = (Signal - b) / a
            # scikit-learn的predict是 y = model.predict(x)
            # 我们训练的是 Signal(y) ~ LogConc(x)。所以反推 x = (y - intercept) / coef
            
            # 为了方便画图，我们直接反解
            slope = reg.coef_[0]
            intercept = reg.intercept_
            pred_log = (signal_val - intercept) / slope
            pred_conc = 10 ** pred_log
            
            # 显示结果
            c1, c2 = st.columns(2)
            c1.metric("Species", pred_species)
            c2.metric("Concentration", f"{pred_conc:.2f} ppb")
            
            # 3. 绘图 (展示异常值剔除效果)
            st.markdown("---")
            st.subheader(f"📈 Calibration Curve (Outliers Removed)")
            
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # A. 画干净数据 (实心点)
            ax.scatter(model_data['X_clean'], model_data['y_clean'], 
                       color='blue', alpha=0.6, label='Valid Samples')
            
            # B. 画异常数据 (红叉) - 如果有的话
            if len(model_data['X_out']) > 0:
                ax.scatter(model_data['X_out'], model_data['y_out'], 
                           color='red', marker='x', s=80, label='Outliers (Discarded)')
            
            # C. 画回归线 (只基于干净数据)
            x_range = np.linspace(min(model_data['X_clean']), max(model_data['X_clean']), 100).reshape(-1, 1)
            y_pred_line = reg.predict(x_range)
            ax.plot(x_range, y_pred_line, 'k--', label='Robust Linear Fit')
            
            # D. 画当前样品
            ax.scatter([pred_log], [signal_val], color='gold', marker='*', s=300, 
                       edgecolors='black', label='Your Sample', zorder=10)
            
            # 标注方程
            r2 = r2_score(model_data['y_clean'], reg.predict(model_data['X_clean']))
            eq = f"y = {slope:.4f}x + {intercept:.4f}\n$R^2$ = {r2:.4f}"
            ax.text(0.05, 0.85, eq, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.9))
            
            ax.set_xlabel("Log10 (Concentration)", fontweight='bold')
            ax.set_ylabel(f"Signal ({model_data['feature']})", fontweight='bold')
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.5)
            
            st.pyplot(fig)
            
            if len(model_data['X_out']) > 0:
                st.caption(f"Note: {len(model_data['X_out'])} outlier points were automatically detected and excluded from the regression.")
