import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ============================================
# 1. 界面配置 (必须放在第一行)
# ============================================
st.set_page_config(
    page_title="Fluo-Sensor AI Pro",
    page_icon="🔬",
    layout="wide",  # 使用宽屏模式，更大气
    initial_sidebar_state="expanded"
)

# 自定义 CSS 美化 (让界面不那么单调)
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    h1 {
        color: #2c3e50;
    }
    .highlight {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. 核心逻辑：加载模型与范围定义
# ============================================
@st.cache_resource
def load_models():
    try:
        return joblib.load('sensor_brain.pkl')
    except:
        return None

models = load_models()

# 定义线性范围 (Linear Range)
# 根据您提供的数据设定
RANGES = {
    'DCP': (0.1, 10000),      # DCP 范围
    'HCl': (4, 400000)        # HCl 范围 (更新到了40万)
}

# ============================================
# 3. 侧边栏：控制面板
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
    st.title("Control Panel")
    st.markdown("---")
    
    st.write("**Input Sensor Signals:**")
    
    # 使用 number_input 微调
    signal_mpa = st.number_input(
        "MPA Signal (Micro-Pit)", 
        value=-0.002500, 
        format="%.6f",
        step=0.0001
    )
    
    signal_flu = st.number_input(
        "Flu Signal (Planar)", 
        value=0.000000, 
        format="%.6f",
        step=0.0001
    )
    
    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Sample", use_container_width=True, type="primary")
    
    st.caption("v2.0 | Powered by SVM & SVR")

# ============================================
# 4. 主界面：仪表盘展示
# ============================================

# 标题区
st.title("🔬 Intelligent Sensor Analysis System")
st.markdown("Real-time identification and quantification of chemical agents.")

if models is None:
    st.error("🚨 Error: Model file 'sensor_brain.pkl' not found. Please run the training script first!")
    st.stop()

# 只有点击按钮才分析
if analyze_btn:
    scaler = models['scaler']
    clf = models['classifier']
    svr_dcp = models['regressor_dcp']
    svr_hcl = models['regressor_hcl']

    # --- A. 数据处理 ---
    input_data = np.array([[signal_mpa, signal_flu]])
    input_scaled = scaler.transform(input_data)
    
    # --- B. 种类识别 ---
    probs = clf.predict_proba(input_scaled)[0]
    max_prob = np.max(probs)
    pred_idx = np.argmax(probs)
    
    # 设定拒识门槛
    CONFIDENCE_THRESHOLD = 0.75
    
    st.markdown("### 📊 Analysis Report")
    
    # 布局：分为两列
    col_left, col_right = st.columns([1, 1.5])

    # === 情况 1: 未知物 ===
    if max_prob < CONFIDENCE_THRESHOLD:
        with col_left:
            st.warning("⚠️ Unknown Substance")
            st.metric("Top Confidence", f"{max_prob*100:.1f}%", delta="- Low", delta_color="inverse")
        with col_right:
            st.info("The signal pattern does not match well with known DCP or HCl profiles.")

    # === 情况 2: 成功识别 ===
    else:
        species = "DCP" if pred_idx == 0 else "HCl"
        
        # --- C. 浓度计算 ---
        if species == 'DCP':
            log_conc = svr_dcp.predict(input_scaled)[0]
            real_conc = 10 ** log_conc
            unit = "ppm" # 假设单位，您可以自己改
            min_limit, max_limit = RANGES['DCP']
        else:
            log_conc = svr_hcl.predict(input_scaled)[0]
            real_conc = 10 ** log_conc
            unit = "ppb" # 假设单位，您可以自己改
            min_limit, max_limit = RANGES['HCl']

        # --- D. 浓度范围判断 ---
        range_status = "Normal"
        status_color = "normal" # normal, off
        
        if real_conc < min_limit:
            range_status = "Below Limit of Detection (LOD)"
            status_color = "off"
            note = f"📉 Calculated value is below the calibrated range (< {min_limit})"
        elif real_conc > max_limit:
            range_status = "Above Linear Range"
            status_color = "off"
            note = f"📈 Calculated value exceeds the calibrated range (> {max_limit})"
        else:
            range_status = "Valid Range"
            status_color = "normal"
            note = "✅ Data falls within the effective linear calibration range."

        # === 界面展示 ===
        with col_left:
            # 展示种类
            st.success(f"Detected: **{species}**")
            # 展示置信度
            st.metric("Identification Confidence", f"{max_prob*100:.2f}%", delta="High Confidence")
        
        with col_right:
            # 展示浓度
            st.metric(f"Predicted Concentration ({unit})", f"{real_conc:.4f}", delta=range_status, delta_color=status_color)
            st.caption(note)
            
            # 进度条可视化 (Log尺度展示)
            st.markdown(f"**Log Scale Position ({species}):**")
            
            # 计算进度条百分比 (把 log_conc 映射到 0~1 之间)
            log_min, log_max = np.log10(min_limit), np.log10(max_limit)
            # 稍微放宽一点进度条边界以便展示
            plot_min, plot_max = log_min - 0.5, log_max + 0.5
            progress = (log_conc - plot_min) / (plot_max - plot_min)
            progress = max(0.0, min(1.0, progress)) # 限制在0-1之间
            
            st.progress(progress)
            st.text(f"Log(Conc): {log_conc:.2f} | Range: [{log_min:.1f} ~ {log_max:.1f}]")

else:
    # 还没点击按钮时的欢迎界面
    st.info("👈 Please enter signal values in the sidebar panel to start analysis.")
    
    # 展示一下参考范围
    with st.expander("ℹ️ View Calibration Ranges"):
        st.table(pd.DataFrame({
            'Species': ['DCP', 'HCl'],
            'Min Conc': [RANGES['DCP'][0], RANGES['HCl'][0]],
            'Max Conc': [RANGES['DCP'][1], RANGES['HCl'][1]]
        }))