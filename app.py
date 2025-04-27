import subprocess
import sys
import os

# التحقق من وجود مكتبة وتثبيتها إذا لم تكن مثبتة
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install package: {package}. Error: {e}")

# تثبيت المكتبات المطلوبة
required_packages = ["streamlit", "Pillow", "ultralytics", "torch"]
for package in required_packages:
    install_package(package)

# التأكد من تثبيت ultralytics قبل استيرادها
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", 'ultralytics'])
    from ultralytics import YOLO

# استيراد المكتبات بعد التأكد من تثبيتها
import streamlit as st
from PIL import Image
import torch

# تحميل الموديل المدرب
model_path = r'C:\GitHub\Streamlit-Yolo-App\yolov8s.pt'  # غير المسار إذا كان موديلك باسم مختلف
model = YOLO(model_path)

# إعداد صفحة Streamlit
st.title('🔍 اكتشاف العناصر في الصور باستخدام YOLO')

uploaded_file = st.file_uploader("ارفع صورة", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file)
    st.image(image, caption='📷 الصورة التي تم رفعها', use_column_width=True)

    # حفظ الصورة مؤقتاً
    temp_path = "temp_uploaded_image.jpg"
    image.save(temp_path)

    # تشغيل التنبؤ
    st.write("🔎 يتم الآن تحليل الصورة...")
    results = model.predict(temp_path)

    # عرض النتائج
    results_image = results[0].plot()
    st.image(results_image, caption='📍 النتائج على الصورة', use_column_width=True)

    # عرض تفاصيل النتائج نصياً
    st.subheader("📋 التفاصيل:")
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        conf = float(box.conf[0])
        st.write(f"- {label} ({conf*100:.2f}%)")

    # حذف الصورة المؤقتة
    os.remove(temp_path)
