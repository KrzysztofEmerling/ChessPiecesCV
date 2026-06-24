import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Konfiguracja i ładowanie danych

CSV_PATH = "train.csv"
IMG_SIZE = 64  # Rozmiar, do którego przeskalujemy wycięte pola szachownicy
NUM_CLASSES = 13  # Klasy od 0 do 12 zgodnie ze słownikiem types

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Nie znaleziono pliku {CSV_PATH}. Uruchom najpierw skrypt generujący CSV.")

df = pd.read_csv(CSV_PATH)

X = []
y = []

print("Ładowanie i przetwarzanie obrazów...")
for idx, row in df.iterrows():
    img_path = row["image_path"]
    label = row["label"]
    
    if os.path.exists(img_path):
        # Odczyt obrazu (BGR -> RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resizing do stałego rozmiaru wejściowego dla CNN
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalizacja pikseli do zakresu [0, 1]
        img = img / 255.0
        
        X.append(img)
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print(f"Załadowano pomyślnie {len(X)} obrazów o kształcie {X.shape[1:]}")


# 2. Podział danych (Dokładnie tak jak w oryginale)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")


# 3. Definicja architektury CNN

model = models.Sequential([
    # Pierwsza warstwa splotowa + pooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    
    # Druga warstwa splotowa + pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Trzecia warstwa splotowa + pooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Spłaszczenie macierzy do wektora i warstwy gęste (Dense)
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),  # Regularyzacja zapobiegająca overfittingowi
    layers.Dense(NUM_CLASSES, activation='softmax')  # Softmax dla klasyfikacji wieloklasowej
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()


# 4. Trening modelu CNN

EPOCHS = 15
BATCH_SIZE = 32

print("\nRozpoczynanie treningu sieci CNN...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1
)


# 5. Ewaluacja i wyniki akuracji

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, val_acc = model.evaluate(X_val, y_val, verbose=0)
_, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*50)
print("WYNIKI AKURACJI (CNN)")
print("="*50)
print(f"Akuracja train: {train_acc:.3f}")
print(f"Akuracja val:   {val_acc:.3f}")
print(f"Akuracja test:  {test_acc:.3f}")

# Predict (Wyciągamy klasę z najwyższym prawdopodobieństwem za pomocą argmax)
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n" + "="*50)
print("CLASSIFICATION REPORT (test)")
print("="*50)
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion matrix:")
print(cm)


# 6. Zapis modelu do pliku (Odpowiednik zapisu wag)

model.save("piece_recognition/model_cnn.h5")
print("\nModel CNN został zapisany do folderu piece_recognition/model_cnn.h5")

# 7. Wizualizacja historii treningu (Krzywe uczenia)

plt.figure(figsize=(12, 5))

# Wykres dokładności (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')
plt.xlabel('Epoka')
plt.ylabel('Dokładność')
plt.title('Dokładność modelu w czasie')
plt.legend()
plt.grid(True)

# Wykres straty (Loss)
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
plt.xlabel('Epoka')
plt.ylabel('Funkcja straty')
plt.title('Strata modelu w czasie')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cnn_training_history.png", dpi=150)
plt.close()
print("Wykres historii uczenia został zapisany jako 'cnn_training_history.png'")