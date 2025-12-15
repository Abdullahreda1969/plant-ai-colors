import os
import numpy as np
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS  # <-- أضف هذا
from PIL import Image
import tensorflow as tf
import io
import sys

# إصلاح الترميز
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

app = Flask(__name__)
CORS(app)  # <-- أضف هذا السطر
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# تحميل نموذج MNIST
print("🚀 بدء تحميل النموذج...")
model = tf.keras.models.load_model('model/mnist_model.keras')
print("✅ النموذج محمل")

# فئات MNIST
CLASS_NAMES = ['صفر', 'واحد', 'اثنان', 'ثلاثة', 'أربعة', 
               'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'لا يوجد ملف'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'لم تحدد ملف'}), 400
    
    # حفظ الملف
    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # معالجة الصورة لـ MNIST
    img = Image.open(filepath).convert('L')  # أبيض/أسود
    img = img.resize((28, 28))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # التنبؤ
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return jsonify({
        'filename': filename,
        'prediction': CLASS_NAMES[predicted_class],
        'confidence': confidence,
        'confidence_percent': round(confidence * 100, 2)
    })

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    print("=" * 50)
    print("🌐 التطبيق جاهز على http://localhost:5000")
    print("🎯 تم تفعيل CORS للاتصال مع المتصفح")
    print("=" * 50)
    app.run(debug=True, port=5000)