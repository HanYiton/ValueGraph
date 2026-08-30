import tensorflow as tf
import numpy as np
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Masking
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, accuracy_score
import json
from numpy.random import seed
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Masking, BatchNormalization

# ثابت کردن تصادفی بودن نتایج
seed(364)
tf.random.set_seed(364)

def labell2strA(label):
    """تبدیل برچسب‌های stance به متن"""
    if label == 0:
        return "support"
    elif label == 1:
        return "comment"
    elif label == 2:
        return "deny"
    elif label == 3:
        return "query"
    else:
        print(label)


def convertsave_competitionformat(idsA, predictionsA, confidenceA):
    """ذخیره نتایج پیش‌بینی‌ها در فرمت رقابتی فقط برای تسک Stance"""
    subtaskaenglish = {}
    
    for i, id in enumerate(idsA):
        subtaskaenglish[id] = labell2strA(predictionsA[i])
    
    # ایجاد مقادیر پیش‌فرض برای تسک Veracity
    subtaskbenglish = {id: ["unverified", 1.0] for id in idsA}
    
    answer = {
        'subtaskaenglish': subtaskaenglish,
        'subtaskbenglish': subtaskbenglish,
        'subtaskadanish': {},
        'subtaskbdanish': {},
        'subtaskarussian': {},
        'subtaskbrussian': {}
    }

    with open("output/answer.json", 'w') as f:
        json.dump(answer, f)
        
    return answer

def load_data():
    """لود داده‌ها از فایل‌های .npy"""
    X_train = np.load('train/train_array.npy')
    y_train_orig = np.load('train/fold_stance_labels.npy')

    train_ids = np.load('train/tweet_ids.npy')

    X_test = np.load('test/test_array.npy')
    y_test_orig = np.load('test/test_labels.npy')
    test_ids = np.load('test/tweet_ids.npy')

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    # اگر برچسب‌ها به صورت one-hot هستند، به برچسب عددی تبدیل می‌کنیم
    if y_train_orig.ndim == 2 and y_train_orig.shape[1] > 1:
        y_train_labels = np.argmax(y_train_orig, axis=1)
        y_test_labels = np.argmax(y_test_orig, axis=1)
    else:
        y_train_labels = y_train_orig
        y_test_labels = y_test_orig
    
    # تبدیل به one-hot encoding با 4 کلاس
    y_train = to_categorical(y_train_labels, num_classes=4)
    y_test = to_categorical(y_test_labels, num_classes=4)
    
    # تغییر شکل برچسب‌ها به (n_samples, 1, 4)
    y_train = y_train.reshape((y_train.shape[0], 1, y_train.shape[1]))
    y_test = y_test.reshape((y_test.shape[0], 1, y_test.shape[1]))
    
    return X_train, y_train, X_test, y_test, train_ids, test_ids, y_train_labels, y_test_labels

def f1_score_metric(y_true, y_pred):
    """محاسبه F1 Score به صورت دستی"""
    # تبدیل پیش‌بینی‌ها و برچسب‌ها به اعداد صحیح
    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)
    
    # محاسبه Precision و Recall
    precision = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    recall = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))
    
    # محاسبه F1 Score
    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1

def build_model(x_train, params):
    """پیاده‌سازی مدل LSTM برای تسک Stance طبق ساختار پیشنهادی با Batch Normalization"""
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
    
    model = Sequential()
    num_features = x_train.shape[2]
    
    # لایه ماسک‌گذاری برای مقادیر صفر
    model.add(Masking(mask_value=0., input_shape=(None, num_features)))
    
    # لایه‌های LSTM
    for nl in range(num_lstm_layers - 1):
        model.add(LSTM(num_lstm_units, 
                      kernel_initializer='glorot_normal',
                      dropout=0.2, 
                      recurrent_dropout=0.2,
                      return_sequences=True))
        # اضافه کردن BatchNormalization پس از LSTM
        #model.add(BatchNormalization())
    
    model.add(LSTM(num_lstm_units, 
                  kernel_initializer='glorot_normal',
                  dropout=0.2, 
                  recurrent_dropout=0.2, 
                  return_sequences=True))
    # اضافه کردن BatchNormalization پس از آخرین LSTM
    #model.add(BatchNormalization())
    
    # لایه‌های متراکم با توزیع زمانی
    for nl in range(num_dense_layers - 1):
        model.add(TimeDistributed(Dense(num_dense_units, activation='relu')))
        model.add(BatchNormalization())
    
    model.add(Dropout(0.2))
    
    # لایه خروجی
    model.add(TimeDistributed(Dense(4, activation='softmax',
                              activity_regularizer=regularizers.l2(l2reg))))
    
    # بهینه‌ساز Adam
    adam = optimizers.Adam(learning_rate=learn_rate, 
                          beta_1=0.9, 
                          beta_2=0.999,
                          epsilon=1e-05)
    
    model.compile(optimizer=adam, 
                 loss='categorical_crossentropy',
                 metrics=["accuracy" , f1_score_metric])
    
    return model
# ذخیره نتایج در فرمت رقابتی
def train_model(model, X_train, y_train, X_val, y_val, params):
    """آموزش مدل"""
    # محاسبه وزن کلاس‌ها برای مدیریت عدم تعادل
    # تبدیل برچسب‌های 3 بعدی به 1 بعدی
    y_labels_flat = np.argmax(y_train, axis=2).flatten()
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_labels_flat), 
                                        y=y_labels_flat)
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)
    
    # ایجاد وزن نمونه‌ها بر اساس وزن کلاس
    sample_weights = np.array([class_weight_dict[label] for label in y_labels_flat])
    sample_weights = sample_weights.reshape((y_train.shape[0], y_train.shape[1]))
    
    # افزودن callback برای کاهش نرخ یادگیری و توقف زودهنگام
    callbacks = [
        #ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, 
        y_train,
        batch_size=params['mb_size'],
        epochs=params['num_epochs'], 
        shuffle=False, 
        validation_data=(X_val, y_val),
        #sample_weight=sample_weights,  # استفاده از sample_weight به جای class_weight
        callbacks=callbacks,
        verbose=1
    )
    return model, history


# پارامترهای بهبود یافته مدل Stance
paramsA = {
    'num_dense_layers':3,
    'num_dense_units': 512,  # کاهش واحدها
    'num_epochs': 100,       # افزایش دوره‌ها
    'num_lstm_units': 500,   # کاهش واحدهای LSTM
    'num_lstm_layers': 3,
    'learn_rate': 1e-4,    # کاهش نرخ یادگیری
    'mb_size': 64,           # کاهش اندازه دسته
    'l2reg': 1e-2,           # افزایش تنظیم‌کننده L2
    'rng_seed': 42
}

# بارگذاری داده‌ها
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# بارگذاری داده‌ها
X_train, y_train, X_test, y_test, train_ids, test_ids, y_train_labels, y_test_labels = load_data()

# تبدیل داده‌ها به ۳ بعدی: (num_samples, timesteps=1, num_features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# استفاده از MinMaxScaler برای نرمال‌سازی داده‌ها
scaler = MinMaxScaler()

# برای نرمال‌سازی داده‌ها، ابتدا باید آن‌ها را به ۲ بعدی تبدیل کنیم (نمونه‌ها، ویژگی‌ها)
# سپس داده‌های ورودی را به ۳ بعدی تبدیل می‌کنیم
X_train_reshaped = X_train.reshape((-1, X_train.shape[-1]))  # تبدیل به (num_samples * timesteps, num_features)
X_test_reshaped = X_test.reshape((-1, X_test.shape[-1]))  # تبدیل به (num_samples * timesteps, num_features)

# نرمال‌سازی داده‌ها
X_train_scaled = scaler.fit_transform(X_train_reshaped)
X_test_scaled = scaler.transform(X_test_reshaped)

# بازگرداندن داده‌ها به شکل ۳ بعدی
X_train = X_train_scaled.reshape((X_train.shape[0], 1, X_train.shape[2]))
X_test = X_test_scaled.reshape((X_test.shape[0], 1, X_test.shape[2]))

# ساخت مدل Stance
modelA = build_model(X_train, paramsA)

# آموزش مدل
modelA, historyA = train_model(modelA, X_train, y_train, X_test, y_test, paramsA)

# پیش‌بینی
pred_probabilities = modelA.predict(X_test, batch_size=paramsA['mb_size'], verbose=1)
confidence = np.max(pred_probabilities, axis=-1)
Y_pred = np.argmax(pred_probabilities, axis=-1)

# تبدیل به شکل 1 بعدی
y_pred_A = Y_pred.flatten()
confidenceA = confidence.flatten()

# ارزیابی مدل Stance
test_acc_A = accuracy_score(y_test_labels, y_pred_A)
f1_A = f1_score(y_test_labels, y_pred_A, average='weighted')

print(f"\nTest accuracy for Stance model: {test_acc_A}")
print(f"Test F1 Score for Stance model: {f1_A}")
