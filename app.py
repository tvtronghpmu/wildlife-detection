import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import time
import os
import datetime
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Hệ Thống Nhận Diện Người và Động Vật Hoang Dã",
   # page_icon=""
    layout="wide"
)

# --- CẤU HÌNH THƯ MỤC ĐẦU RA ---
IMAGE_OUTPUT_FOLDER = "Image_output"
VIDEO_OUTPUT_FOLDER = "Video_output"

# Tạo các thư mục nếu chưa tồn tại
for folder in [IMAGE_OUTPUT_FOLDER, VIDEO_OUTPUT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# --- CSS TÙY CHỈNH ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; }
    .stat-box {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 1px 1px 5px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    /* --- TÙY CHỈNH CON TRỎ CHUỘT --- */
    [data-testid="stSidebar"] div[data-baseweb="select"] { cursor: pointer; }
    [data-testid="stSidebar"] div[data-baseweb="select"]:hover { border-color: #ff4b4b; }
    </style>
    """, unsafe_allow_html=True)

# --- HÀM LOAD MODEL ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

# --- SIDEBAR: CẤU HÌNH ---
with st.sidebar:
    st.title("⚙️ Bảng Điều Khiển")
    
    app_mode = st.selectbox("Chọn chế độ:", ["📸 Nhận diện Hình ảnh", "🎥 Nhận diện Video"])
    st.markdown("---")
    
    st.subheader("Model Config")
    model_source = st.radio("Nguồn Model:", ("Mặc định (best.pt)", "Upload (.pt)"))
    model_path = "best.pt"
    if model_source == "Upload (.pt)":
        uploaded_model = st.file_uploader("Upload model file", type=["pt"])
        if uploaded_model:
            with open("temp_model.pt", "wb") as f:
                f.write(uploaded_model.getbuffer())
            model_path = "temp_model.pt"
    
    st.subheader("Tham số dự đoán")
    conf_threshold = st.slider("Độ tin cậy (Confidence)", 0.0, 1.0, 0.4, 0.05)
    iou_threshold = st.slider("Ngưỡng chồng lấp (IOU)", 0.0, 1.0, 0.45, 0.05)

# --- MAIN APP ---
st.title("Hệ thống nhận dạng đối tượng (người, động vật hoang dã) sử dụng YOLOv11s")

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Chưa tìm thấy model! Hãy đảm bảo file '{model_path}' nằm cùng thư mục.")
    st.stop()

# ==========================================
# CHẾ ĐỘ 1: XỬ LÝ HÌNH ẢNH
# ==========================================
if app_mode == "📸 Nhận diện Hình ảnh":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Input Image")
        uploaded_file = st.file_uploader("Tải ảnh lên...", type=['jpg', 'png', 'jpeg'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Ảnh gốc", use_container_width=True)
            
            if st.button("🔍 Phân tích ngay", type="primary"):
                with col2:
                    st.subheader("2. Kết quả & Thống kê")
                    with st.spinner('Đang xử lý...'):
                        results = model.predict(image, conf=conf_threshold, iou=iou_threshold)
                        res_plotted = results[0].plot()[:, :, ::-1] 
                        
                        st.image(res_plotted, caption="Kết quả nhận diện", use_container_width=True)
                        
                        # --- LƯU ẢNH ---
                        try:
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            save_path = os.path.join(IMAGE_OUTPUT_FOLDER, f"result_{timestamp}.jpg")
                            Image.fromarray(res_plotted).save(save_path)
                            st.success(f"✅ Đã lưu ảnh vào: `{save_path}`")
                        except Exception as e:
                            st.error(f"Lỗi lưu ảnh: {e}")
                        
                        # Thống kê
                        detections = results[0].boxes.cls.cpu().numpy()
                        class_names = results[0].names
                        if len(detections) > 0:
                            counts = {}
                            for det in detections:
                                name = class_names[int(det)]
                                counts[name] = counts.get(name, 0) + 1
                            st.write("### 📊 Số lượng phát hiện:")
                            for name, count in counts.items():
                                st.markdown(f"""<div class="stat-box"><b>{name.upper()}:</b> {count} cá thể</div>""", unsafe_allow_html=True)
                        else:
                            st.warning("Không phát hiện vật thể nào.")

# ==========================================
# CHẾ ĐỘ 2: XỬ LÝ VIDEO
# ==========================================
elif app_mode == "🎥 Nhận diện Video":
    st.subheader("Upload Video để phân tích thời gian thực")
    uploaded_video = st.file_uploader("Chọn video (mp4, avi, mov)...", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        col_video, col_stat = st.columns([3, 1])
        
        with col_video:
            st.markdown("**Preview Xử lý:**")
            st_frame = st.empty()
            
        with col_stat:
            st.markdown("**Trạng thái:**")
            kpi_text = st.empty()
            btn_start = st.button("▶️ Bắt đầu chạy")

        if btn_start:
            cap = cv2.VideoCapture(video_path)
            
            # Lấy thông số video gốc
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # --- CẤU HÌNH GHI VIDEO (OUTPUT) ---
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_video_path = os.path.join(VIDEO_OUTPUT_FOLDER, f"video_{timestamp}.mp4")
            
            # Định dạng codec (mp4v cho file .mp4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            kpi_text.info(f"Độ phân giải: {width}x{height} | FPS: {fps}")
            prev_time = 0
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Tính FPS xử lý
                    curr_time = time.time()
                    fps_proc = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
                    prev_time = curr_time
                    
                    # Dự đoán
                    results = model.predict(frame, conf=conf_threshold, iou=iou_threshold)
                    
                    # Vẽ box (kết quả là BGR chuẩn OpenCV)
                    res_plotted = results[0].plot()
                    
                    # 1. Hiển thị lên Web (Cần convert sang RGB)
                    st_frame.image(res_plotted, channels="BGR", caption=f"Processing FPS: {int(fps_proc)}", use_container_width=True)
                    
                    # 2. Ghi vào file Video (Giữ nguyên BGR)
                    out.write(res_plotted)
                    
            except Exception as e:
                st.error(f"Có lỗi xảy ra: {e}")
            finally:
                # Giải phóng tài nguyên
                cap.release()
                out.release()
                
            st.success("✅ Đã xử lý xong video!")
            st.success(f"📁 Video đã được lưu tại: `{output_video_path}`")