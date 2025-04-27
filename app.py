import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
import os

# تحميل الموديل المدرب
model_path = 'yolov8s.pt'  # لازم يكون الملف مرفوع مع ملفات المشروع أو يتم تحميله
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
