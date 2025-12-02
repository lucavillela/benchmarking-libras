import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- CONFIGURAÇÕES ---
DATASET_PATH = "aumentado"  # Para ler os nomes das classes

# IMPORTANTE: Definir tamanhos diferentes conforme o treinamento de cada um
IMG_SIZE_TAMIRES = 128 # O modelo CNN foi treinado com 128
IMG_SIZE_YOLO = 224    # O YOLO geralmente usa 224

# 1. Carregar Nomes das Classes
try:
    CLASS_NAMES = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])
    print(f"Classes carregadas: {CLASS_NAMES}")
except:
    print("ERRO: Pasta do dataset não encontrada. Defina as classes manualmente.")
    # Exemplo de fallback se a pasta não existir
    CLASS_NAMES = ['A', 'B', 'C', 'D', 'E'] 

# =========================================================
# CARREGAMENTO DOS MODELOS
# =========================================================

print("\n--- Carregando Modelos... ---")

# 1. Modelo Tamires (Keras CNN)
try:
    model_tamires = load_model('modelo_tamires.keras')
    print("✅ Modelo Tamires carregado.")
except Exception as e:
    print(f"❌ Modelo Tamires não encontrado: {e}")
    model_tamires = None

# 2. Modelo Gabriel (Keras MLP + MediaPipe)
try:
    model_gabriel = load_model('modelo_gabriel.keras')
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    print("✅ Modelo Gabriel carregado.")
except Exception as e:
    print(f"❌ Modelo Gabriel não encontrado: {e}")
    model_gabriel = None

# 3. Modelo Márcio (YOLOv5 PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    # Recriar a arquitetura
    model_marcio = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s-cls.pt', force_reload=False)
    
    # Lógica de adaptação da camada final (Head) igual ao script de treino
    model_sequence = model_marcio.model
    final_module = model_sequence.model[-1]
    
    # Verifica se é Linear ou Sequential e substitui
    linear_layer = None
    if isinstance(final_module, nn.Linear):
        linear_layer = final_module
    else:
        for m in final_module.modules():
            if isinstance(m, nn.Linear):
                linear_layer = m
                break
                
    if linear_layer:
        in_features = linear_layer.in_features
        # Substitui a camada final
        if isinstance(final_module, nn.Linear):
            model_sequence.model[-1] = nn.Linear(in_features, len(CLASS_NAMES))
        else:
            # Se for um bloco, tenta adicionar/substituir (lógica genérica)
            # Para simplificar na interface, assumimos que o state_dict vai sobrescrever corretamente
            # desde que a arquitetura 'bata'. Vamos recriar a linear:
            final_module.linear = nn.Linear(in_features, len(CLASS_NAMES))

    # Carregar os pesos treinados
    model_marcio.load_state_dict(torch.load('modelo_marcio_weights.pth', map_location=device))
    model_marcio.to(device)
    model_marcio.eval()
    print("✅ Modelo Márcio carregado.")
except Exception as e:
    print(f"❌ Modelo Márcio com erro (verifique se treinou e salvou .pth): {e}")
    model_marcio = None

# =========================================================
# FUNÇÕES DE PREDIÇÃO
# =========================================================

def predict_tamires(frame):
    if model_tamires is None: return "Erro Modelo", 0.0
    
    # Redimensiona para 128x128 (Tamires)
    img = cv2.resize(frame, (IMG_SIZE_TAMIRES, IMG_SIZE_TAMIRES))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    preds = model_tamires.predict(img, verbose=0)
    idx = np.argmax(preds)
    conf = np.max(preds)
    return CLASS_NAMES[idx], conf

def predict_gabriel(frame):
    if model_gabriel is None: return "Erro Modelo", 0.0
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)
    
    if results.multi_hand_landmarks:
        # Desenha o esqueleto na tela
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            
        landmarks = []
        for lm in results.multi_hand_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y])
        
        input_data = np.array([landmarks])
        preds = model_gabriel.predict(input_data, verbose=0)
        idx = np.argmax(preds)
        conf = np.max(preds)
        return CLASS_NAMES[idx], conf
    else:
        return "Nenhuma mão", 0.0

def predict_marcio(frame):
    if model_marcio is None: return "Erro Modelo", 0.0
    
    # Redimensiona para 224x224 (YOLO)
    img = cv2.resize(frame, (IMG_SIZE_YOLO, IMG_SIZE_YOLO))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.transpose(img, (2, 0, 1)) # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    
    tensor = torch.from_numpy(img).to(device)
    
    with torch.no_grad():
        preds = model_marcio(tensor)
        probs = torch.nn.functional.softmax(preds, dim=1)
        conf, idx = torch.max(probs, 1)
        
    return CLASS_NAMES[idx.item()], conf.item()

# =========================================================
# LOOP PRINCIPAL (WEBCAM)
# =========================================================

cap = cv2.VideoCapture(0)
current_model = 1 

print("\n--- INICIANDO WEBCAM ---")
print("TECLAS DE CONTROLE:")
print("[1] CNN (Tamires) | [2] MediaPipe (Gabriel) | [3] YOLOv5 (Márcio) | [Q] Sair")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Flip horizontal (espelho)
    frame = cv2.flip(frame, 1)
    
    label = "..."
    conf = 0.0
    model_name = ""
    color = (255, 255, 255)
    
    try:
        if current_model == 1:
            model_name = "Tamires (CNN 128px)"
            color = (255, 0, 0) # Azul
            label, conf = predict_tamires(frame)
            
        elif current_model == 2:
            model_name = "Gabriel (Esqueleto)"
            color = (0, 255, 0) # Verde
            label, conf = predict_gabriel(frame) 
            
        elif current_model == 3:
            model_name = "Marcio (YOLOv5 224px)"
            color = (0, 0, 255) # Vermelho
            label, conf = predict_marcio(frame)
    except Exception as e:
        label = "Erro Pred"
        print(f"Erro na predição: {e}")

    # Visualização
    cv2.rectangle(frame, (0, 0), (640, 90), (0, 0, 0), -1) 
    
    cv2.putText(frame, f"Modelo: {model_name}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    if label == "Nenhuma mão":
        text_res = "Mao nao detectada"
    else:
        text_res = f"Letra: {label} ({conf*100:.1f}%)"
        
    cv2.putText(frame, text_res, (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.imshow('Interface Benchmarking Libras', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('1'): current_model = 1
    if key == ord('2'): current_model = 2
    if key == ord('3'): current_model = 3

cap.release()
cv2.destroyAllWindows()