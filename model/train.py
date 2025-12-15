import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

print("=" * 60)
print("🌱 بدء تدريب نموذج التعرف على الأرقام (MNIST)")
print("=" * 60)

# 1. تحميل بيانات MNIST
print("📥 جاري تحميل البيانات...")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 2. إعادة تشكيل الأبعاد وتطبيع
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

print(f"✅ تم تحميل {len(train_images)} صورة تدريب و {len(test_images)} صورة اختبار")
print(f"📐 أبعاد الصور: {train_images.shape[1:]} (ارتفاع×عرض×قنوات)")

# 3. تعريف الفئات
class_names = ['صفر', 'واحد', 'اثنان', 'ثلاثة', 'أربعة', 
               'خمسة', 'ستة', 'سبعة', 'ثمانية', 'تسعة']
print(f"🔤 الفئات: {class_names}")

# 4. تحويل إلى Dataset
BATCH_SIZE = 32
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(60000).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
val_ds = val_ds.batch(BATCH_SIZE)

print(f"📊 حجم الدفعة: {BATCH_SIZE} صورة")

# 5. بناء النموذج - مهم: لصور MNIST (28,28,1)
print("\n🧠 بناء نموذج الشبكة العصبية...")
model = models.Sequential([
    # الطبقة الأولى مهمة: input_shape يجب أن يكون (28, 28, 1)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 فئات للأرقام 0-9
])

# 6. تجميع النموذج
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 7. عرض ملخص النموذج
print("\n📋 ملخص النموذج:")
model.summary()

# 8. التدريب
EPOCHS = 5  # قللنا لسرعة التدريب
print(f"\n🔥 بدء التدريب ({EPOCHS} دورات)...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# 9. التقييم
print("\n📈 تقييم النموذج على بيانات الاختبار...")
test_loss, test_acc = model.evaluate(val_ds)
print(f"✅ الدقة النهائية: {test_acc:.4f} ({test_acc*100:.1f}%)")

# 10. حفظ النموذج
model_path = 'mnist_model.keras'
model.save(model_path)
print(f"💾 النموذج محفوظ في: {model_path}")

# 11. اختبار تنبؤ عشوائي
print("\n🧪 اختبار تنبؤ عشوائي...")
sample_image = test_images[0:1]  # أول صورة اختبار
predictions = model.predict(sample_image)
predicted_class = np.argmax(predictions[0])
confidence = np.max(predictions[0])
print(f"📸 الصورة الأولى من الاختبار:")
print(f"   - التنبؤ: {class_names[predicted_class]}")
print(f"   - الثقة: {confidence*100:.1f}%")
print(f"   - التصنيف الحقيقي: {class_names[test_labels[0]]}")

print("\n" + "=" * 60)
print("🎉 اكتمل التدريب بنجاح!")
print("=" * 60)