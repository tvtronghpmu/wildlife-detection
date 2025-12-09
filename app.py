import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import time
import os
import datetime
import numpy as np

# ==========================================
# 1. CẤU HÌNH TRANG (PAGE CONFIG)
# ==========================================
st.set_page_config(
    page_title="Human and animal detection with AI",
    page_icon="🐾", # Icon dấu chân (Tracking)
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CẤU HÌNH THƯ MỤC ĐẦU RA ---
IMAGE_OUTPUT_FOLDER = "Image_output"
VIDEO_OUTPUT_FOLDER = "Video_output"
for folder in [IMAGE_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ==========================================
# 2. CSS TÙY CHỈNH (PROFESSIONAL UI)
# ==========================================
st.markdown("""
    <style>
    /* 1. RESET PADDING */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
    }
    
    /* 2. BANNER KỸ THUẬT SỐ */
    .custom-banner {
        width: 100%;
        height: 220px;
        overflow: hidden;
        border-radius: 0px 0px 15px 15px; /* Bo tròn 2 góc dưới */
        box-shadow: 0 10px 20px rgba(0,0,0,0.15); /* Bóng đổ sâu hơn */
        position: relative;
    }
    .custom-banner img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 30%; /* Căn chỉnh vị trí ảnh */
        filter: brightness(0.9); /* Làm tối nhẹ ảnh để chữ nổi hơn nếu có */
    }

    /* 3. SIDEBAR HIỆN ĐẠI */
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        background-color: #f4f6f9; /* Màu xám xanh công nghệ */
        border-right: 1px solid #e0e0e0;
    }
    
    /* 4. BUTTON CHUYÊN NGHIỆP */
    div.stButton > button {
        background: linear-gradient(45deg, #1b5e20, #2e7d32); /* Gradient xanh quân đội */
        color: white;
        border-radius: 6px;
        height: 55px;
        font-size: 16px;
        font-family: 'Roboto Mono', monospace; /* Font kỹ thuật */
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 10px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
        text-transform: uppercase; /* Chữ in hoa */
        letter-spacing: 1px;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.4);
    }

    /* 5. METRIC CARD (THẺ SỐ LIỆU) */
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1b5e20;
        border-bottom: 2px solid #f0f0f0;
        text-align: center;
        margin-bottom: 10px;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.02);
    }
    .metric-value { 
        font-size: 26px; 
        font-weight: 800; 
        color: #1b5e20; 
        font-family: 'Segoe UI', sans-serif;
    }
    .metric-label { 
        font-size: 11px; 
        color: #555; 
        text-transform: uppercase; 
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* 6. HEADER CHỮ */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        color: #2c3e50;
    }
    
    /* Con trỏ chuột */
    [data-testid="stSidebar"] [data-baseweb="select"] { cursor: pointer !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] * { cursor: pointer !important; }
    
    /* Ẩn footer */
    footer, header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

# ==========================================
# 3. SIDEBAR (THANH ĐIỀU KHIỂN)
# ==========================================
with st.sidebar:
    # Header Sidebar với icon Đại bàng (Tầm nhìn)
    st.markdown("<h2 style='text-align: center; color: #1b5e20; margin-bottom: 5px;'>🦅 Trần Văn Trọng <br>Nguyễn Thanh Hà</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 12px;'><i>Human and animal detection with AI System v1.0</i></p>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.subheader("🎛️ Bảng Điều Khiển")
    
    # Selectbox với icon mô tả chức năng
    app_mode = st.selectbox(
        "Chọn chế độ quét:",
        ["🖼️ Phân Tích Hình Ảnh (Image)", "📡 Giám Sát Video (Live Stream)"],
        index=0
    )
    
    st.markdown("---")
    
    with st.expander("🛠️ Thiết Lập Kỹ Thuật", expanded=True):
        model_source = st.radio("Core Model:", ("Standard (best.pt)", "Custom Upload (.pt)"))
        model_path = "best.pt"
        if model_source == "Custom Upload (.pt)":
            uploaded_model = st.file_uploader("Upload weights", type=["pt"])
            if uploaded_model:
                with open("temp_model.pt", "wb") as f:
                    f.write(uploaded_model.getbuffer())
                model_path = "temp_model.pt"
        
        st.markdown("#### 🎚️ Bộ Lọc Tín Hiệu")
        conf_threshold = st.slider("Độ nhạy (Confidence)", 0.0, 1.0, 0.45, 0.05)
        iou_threshold = st.slider("Ngưỡng chồng lấp (NMS)", 0.0, 1.0, 0.45, 0.05)

# ==========================================
# 4. MAIN INTERFACE
# ==========================================

# --- BANNER ẢNH RỪNG (CỐ ĐỊNH 220PX) ---
# Link ảnh chất lượng cao chủ đề Rừng/Núi
BANNER_URL = "https://images.unsplash.com/photo-1470071459604-3b5ec3a7fe05?q=80&w=2000&auto=format&fit=crop"

st.markdown(f"""
    <div class="custom-banner">
        <img src="{BANNER_URL}">
    </div>
""", unsafe_allow_html=True)

# Load Model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"⚠️ SYSTEM ERROR: Model not found at '{model_path}'.")
    st.stop()

# Header chính
# Tách tên mode để hiển thị đẹp hơn
mode_title = "IMAGERY INTELLIGENCE" if "Image" in app_mode else "VIDEO SURVEILLANCE"
mode_icon = "📸" if "Image" in app_mode else "🎥"

st.markdown(f"""
    <div style="text-align: center; margin-top: 20px;">
        <h1 style="color: #1b5e20; margin-bottom: 0px;">{mode_icon} {mode_title}</h1>
        <p style="color: #555; font-size: 16px;">Hệ thống nhận diện người và một số loài động vật </p>
    </div>
    <hr style="border-top: 1px solid #ddd; margin-bottom: 30px;">
""", unsafe_allow_html=True)

# --- MODE 1: HÌNH ẢNH ---
if "Image" in app_mode:
    col_input, col_result = st.columns([4, 6], gap="large")
    
    with col_input:
        st.markdown("### 📥 Dữ Liệu Đầu Vào")
        uploaded_file = st.file_uploader("Tải ảnh định dạng JPG/PNG...", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Source Image", use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⚡ KÍCH HOẠT QUÉT AI", type="primary"):
                with col_result:
                    st.markdown("### 🎯 Kết Quả Phân Tích")
                    with st.spinner('🔄 Đang xử lý thuật toán...'):
                        results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
                        res_plotted = results[0].plot()[:, :, ::-1]
                        
                        st.image(res_plotted, caption="AI Detected Overlay", use_container_width=True)
                        
                        # Save
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"detect_{timestamp}.jpg")
                        Image.fromarray(res_plotted).save(save_path)
                        st.success(f"💾 Dữ liệu đã lưu: `{save_path}`")

                        # Dashboard
                        st.markdown("### 📋 Báo Cáo Số Liệu")
                        detections = results[0].boxes.cls.cpu().numpy()
                        class_names = results[0].names
                        
                        if len(detections) > 0:
                            counts = {}
                            for det in detections:
                                name = class_names[int(det)]
                                counts[name] = counts.get(name, 0) + 1
                            
                            cols = st.columns(3)
                            idx = 0
                            for name, count in counts.items():
                                with cols[idx % 3]:
                                    st.markdown(f"""
                                        <div class="metric-card">
                                            <div class="metric-value">{count}</div>
                                            <div class="metric-label">{name}</div>
                                        </div>
                                    """, unsafe_allow_html=True)
                                idx += 1
                        else:
                            st.info("ℹ️ Không phát hiện mục tiêu trong vùng quét.")

# --- MODE 2: VIDEO ---
elif "Video" in app_mode:
    st.markdown("### 📥 Nguồn Tín Hiệu Video")
    uploaded_video = st.file_uploader("Tải video MP4/AVI...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        c1, c2 = st.columns([3, 1])
        
        with c1:
            st.markdown("**📺 Màn Hình Giám Sát**")
            st_frame = st.empty()
            
        with c2:
            st.markdown("**📡 Trạng Thái Hệ Thống**")
            kpi_fps = st.empty()
            kpi_res = st.empty()
            st.markdown("<br>", unsafe_allow_html=True)
            btn_start = st.button("▶️ BẮT ĐẦU GIÁM SÁT")

        if btn_start:
            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_input = int(cap.get(cv2.CAP_PROP_FPS))
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(VIDEO_OUTPUT_FOLDER, f"surveillance_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_video_path, fourcc, fps_input, (width, height))
            
            kpi_res.info(f"Res: {width}x{height} px")
            prev_time = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                curr_time = time.time()
                fps_proc = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                prev_time = curr_time
                
                results = model.predict(frame, conf=conf_threshold, iou=iou_threshold)
                res_plotted = results[0].plot()
                
                st_frame.image(res_plotted, channels="BGR", use_container_width=True)
                
                # Card FPS chuyên nghiệp hơn
                kpi_fps.markdown(f"""
                <div class="metric-card" style="padding: 10px; border-left: 4px solid #d32f2f;">
                    <div class="metric-value" style="color: #d32f2f;">{int(fps_proc)}</div>
                    <div class="metric-label">REAL-TIME FPS</div>
                </div>
                """, unsafe_allow_html=True)
                
                out.write(res_plotted)
                
            cap.release()
            out.release()
            st.balloons()
            st.success(f"✅ Phiên giám sát kết thúc. Video lưu tại: `{output_video_path}`")
