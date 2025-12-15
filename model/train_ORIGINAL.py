# نصيحة: train.py
# تدريب نموذج بسيط لتصنيف صور النباتات (سليم/مريض)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# 1. تحديد مسارات البيانات
dataset_dir = "../dataset"  # المسار النسبي لمجلد dataset
img_height = 180  # ارتفاع الصورة بعد تغيير حجمها
img_width = 180   # عرض الصورة بعد تغيير حجمها
batch_size = 2    # حجم الدفعة (صغير لأن بياناتنا قليلة)
epochs = 5        # عدد دورات التدريب (قليل للتجربة السريعة)

# 2. تحميل البيانات وتقسيمها باستخدام ImageDataGenerator
# هذا الكلاس يقوم تلقائياً بقراءة الصور من المجلدات وتصنيفها حسب اسم المجلد
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,        # نستخدم 20% من البيانات للتحقق (validation)
    subset="training",           # هذه هي مجموعة التدريب
    seed=123,                    # بذرة عشوائية للتكرارية
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    subset="validation",         # هذه هي مجموعة التحقق
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# أسماء الفئات (ستؤخذ تلقائياً من أسماء المجلدات)
class_names = train_ds.class_names
print("الفئات المكتشفة:", class_names)  # يجب أن يطبع: ['diseased', 'healthy']

# 3. بناء نموذج تسلسلي بسيط
model = keras.Sequential([
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),  # تسوية قيم البكسل
    layers.Conv2D(16, 3, padding='same', activation='relu'),           # طبقة تلافيفية
    layers.MaxPooling2D(),                                              # طبقة تجميع
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),                                                   # تسطيح المخرجات
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names))                                      # طبقة الإخراج (عقدتين)
])

# 4. تجميع النموذج
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 5. عرض ملخص النموذج
model.summary()

# 6. تدريب النموذج
print("بدء التدريب...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# 7. حفظ النموذج المدرب
model.save("plant_disease_model.keras")  # يمكن استخدام .h5 أيضاً
print("✅ النموذج المدرب تم حفظه كـ 'plant_disease_model.keras' في مجلد 'model/'")

# 8. تقييم النموذج على بيانات التحقق
loss, accuracy = model.evaluate(val_ds)
print(f"الخسارة النهائية على بيانات التحقق: {loss:.4f}")
print(f"الدقة النهائية على بيانات التحقق: {accuracy:.4f}")