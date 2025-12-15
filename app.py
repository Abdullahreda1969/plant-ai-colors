# ملف: app/app.py
# تطبيق Flask الأساسي لتطبيق plant-scanner-3

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from flask import Flask, jsonify, request, render_template
from werkzeug.utils import secure_filename
from datetime import datetime

# تحميل نموذج الذكاء الاصطناعي (مع المسار الصحيح)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # مجلد app/
MODEL_PATH = os.path.join(BASE_DIR, "model", "mnist_model.keras")
print(f"جاري تحميل النموذج من: {MODEL_PATH}")

try:
    ai_model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ النموذج تم تحميله بنجاح في الذاكرة")
except Exception as e:
    print(f"❌ فشل تحميل النموذج: {e}")
    ai_model = None

# أسماء الفئات
class_names = ['صفر', 'واحد', 'اثنان', 'ثلاثة', 'أربعة', 
               'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة']
# 1. تهيئة تطبيق Flask
app = Flask(__name__, 
            template_folder=os.path.join(BASE_DIR, '..', 'templates'))
@app.route('/index')
@app.route('/home')
def index():
    """عرض واجهة المستخدم الرئيسية"""
    return render_template('index.html')

@app.route('/')
def home():
    """إعادة توجيه إلى الواجهة الرئيسية"""
    return index()

# 3. نقطة نهاية للتحقق من صحة الخادم (للمراقبة)
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'service': 'plant-scanner-3-backend'})

# 4. نقطة نهاية رفع الصورة (نهائية مع الذكاء الاصطناعي)
@app.route('/upload', methods=['POST'])
def upload_image():
    """
    نقطة نهاية لرفع صورة نبات.
    تقبل صورة في حقل اسمه 'file'.
    تحفظها مؤقتاً ثم تعالجها بالنموذج وتُرجع التنبؤ.
    """
    # 1. التحقق من وجود ملف في الطلب
    if 'file' not in request.files:
        return jsonify({'error': 'لا يوجد ملف في الطلب'}), 400
    
    file = request.files['file']
    
    # 2. التحقق من أن المستخدم اختار ملفاً
    if file.filename == '':
        return jsonify({'error': 'لم يتم اختيار ملف'}), 400
    
    # 3. التحقق من نوع الملف
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    if not ('.' in file.filename and 
            file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
        return jsonify({'error': 'نوع الملف غير مسموح به'}), 400
    
    # 4. إنشاء اسم آمن للملف وحفظه
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_filename = secure_filename(file.filename)
    filename = f"{timestamp}_{original_filename}"
    filepath = os.path.join('uploads', filename)
    
    file.save(filepath)
    
    # ===== معالجة الصورة بالذكاء الاصطناعي =====
    prediction_result = None
    if ai_model is not None:
        try:
            # يجب أن تغير الصورة إلى 28x28 أبيض وأسود
            img = Image.open(image_path).convert('L')  # 'L' = أبيض وأسود
            img = img.resize((28, 28))
            img_array = np.array(img) / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)  # أبعاد MNIST            
            # ج. التنبؤ باستخدام النموذج
            predictions = ai_model.predict(img_array, verbose=0)
            score = tf.nn.softmax(predictions[0])
            
            # د. تفسير النتائج
            predicted_class = CLASS_NAMES[np.argmax(score)]
            confidence = float(np.max(score)) * 100
            
            prediction_result = {
                'class': predicted_class,
                'confidence': round(confidence, 2),
                'class_arabic': 'مريضة' if predicted_class == 'diseased' else 'سليمة',
                'probabilities': {
                    'diseased': float(score[0]),
                    'healthy': float(score[1])
                }
            }
            
        except Exception as e:
            prediction_result = {'error_in_processing': str(e)}
    else:
        prediction_result = {'error': 'النموذج غير متاح للتنبؤ'}
    
    # 5. إرجاع الرد مع نتيجة التنبؤ
    return jsonify({
        'message': 'تم استلام ومعالجة الصورة بنجاح',
        'filename': filename,
        'filepath': filepath,
        'prediction': prediction_result
    }), 200

# 5. تشغيل التطبيق (فقط إذا تم تنفيذ الملف مباشرة)
if __name__ == '__main__':
    # التطبيق سيعمل على http://localhost:5000
    app.run(debug=True, host='0.0.0.0', port=5000)