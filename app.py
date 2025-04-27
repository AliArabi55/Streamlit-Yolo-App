import subprocess
import sys
import os

# التحقق من وجود ملف requirements.txt وتثبيت المكتبات المطلوبة
def install_requirements():
    if not os.path.isfile('requirements.txt'):
        print("لم يتم العثور على ملف requirements.txt!")
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 'requirements.txt'])

# تثبيت المتطلبات إذا كانت غير مثبتة
install_requirements()

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
