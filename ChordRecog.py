import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 確保 TensorFlow 有抓到 M2 的 GPU (Metal)
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

def extract_chroma_windows(audio_path, window_size=11):
    """
    讀取音訊並轉換為 (樣本數, 12, 11, 1) 的 CNN 輸入格式
    """
    print(f"正在處理音訊: {audio_path}")
    # 1. 讀取音訊檔 (預設取樣率 22050Hz)
    y, sr = librosa.load(audio_path, sr=22050)

    # 2. 擷取 CQT Chroma 特徵 (精準對應 12 平均律)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Chroma shape 目前是 (12 個半音, 時間幀數)

    # 3. 準備滑動視窗
    half_window = window_size // 2
    features = []
    
    # 在時間軸前後補零 (Padding)，確保首尾的幀也能捕捉到完整的 11 幀視窗
    chroma_padded = np.pad(chroma, ((0, 0), (half_window, half_window)), mode='constant')

    for i in range(chroma.shape[1]):
        # 沿著時間軸，每次切出 12(高) x 11(寬) 的特徵圖
        window = chroma_padded[:, i : i + window_size]
        features.append(window)

    # 4. 轉換形狀以符合 CNN 格式
    features = np.array(features)                 # (時間幀數, 12, 11)
    features = np.expand_dims(features, axis=-1)  # 增加 Channel 維度 -> (時間幀數, 12, 11, 1)

    return features

def build_chord_cnn(num_classes=4):
    """
    建立針對 12x11 頻譜特徵設計的 2D CNN 模型
    """
    model = models.Sequential([
        # 第一層卷積
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(12, 11, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # 第二層卷積
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # 展平與全連接層
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        
        # 輸出層
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # ==========================================
    # 測試管線 (使用你自己的吉他音訊檔替換這裡)
    # ==========================================
    # 假設我們有一個包含 4 個基本和弦 (C, G, Am, F) 的音訊檔
    dummy_audio_path = librosa.ex('trumpet') # 這裡暫時用 librosa 內建音訊測試管線是否暢通
    
    # 1. 擷取特徵
    X = extract_chroma_windows(dummy_audio_path, window_size=11)
    print(f"特徵擷取完成！輸入形狀為: {X.shape}") 
    # 預期輸出類似: (時間幀數, 12, 11, 1)

    # 2. 建立假標籤 (這部分未來需要你針對音訊時間軸手動或自動標記)
    # 這裡我們隨機生成 0~3 的標籤來代表 4 種和弦，長度等於時間幀數
    num_frames = X.shape[0]
    y = np.random.randint(0, 4, size=num_frames) 

    # 3. 切割訓練集與測試集 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 建立與訓練模型
    print("開始建構模型...")
    model = build_chord_cnn(num_classes=4)
    model.summary()

    print("開始訓練...")
    history = model.fit(
        X_train, y_train,
        epochs=10,            # 測試階段先跑 10 輪
        batch_size=32,        # M2 記憶體夠大，未來資料多時可以開到 64 或 128
        validation_data=(X_test, y_test)
    )