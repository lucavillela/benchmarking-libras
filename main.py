import albumentations as A
import cv2
import os
import numpy as np
from PIL import Image
import pillow_heif 

# --- REGISTRA O PLUGIN HEIF NO PILLOW ---
pillow_heif.register_heif_opener()
# ----------------------------------------

# --- CONFIGURAÇÃO ---
INPUT_DIR = 'alfabeto_libras_dataset' 
OUTPUT_DIR = 'aumentado'
N_COPIES = 5 
TARGET_SIZE = 512  # NOVO: Definição da resolução alvo (512x512)
# --------------------


# 3. Definição do Pipeline de Aumentação (FINAL)
pipeline_aumentacao = A.Compose([
    # --- Geométricas ---
    A.HorizontalFlip(p=0.8),
    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=10,
        p=0.7
    ),
    
    # --- Cores e Iluminação ---
    A.RandomBrightnessContrast(
        brightness_limit=0.2, 
        contrast_limit=0.2,   
        p=0.6
    ),
    A.ColorJitter(
        brightness=0, contrast=0,
        saturation=0.2,           
        hue=0.05,                 
        p=0.5
    ),

    # --- Ruído e Oclusão ---
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    
    # --- REDIMENSIONAMENTO (RESIZE) ---
    # Colocamos como a última transformação para garantir que todas as
    # imagens de saída tenham 512x512 pixels.
    A.Resize(height=TARGET_SIZE, width=TARGET_SIZE, p=1.0) # p=1.0 garante que sempre será aplicado
])


# 4. Função Principal de Aumentação em Lote (ADAPTADA)
def apply_augmentation_batch():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for class_name in os.listdir(INPUT_DIR):
        class_path = os.path.join(INPUT_DIR, class_name)
        
        if not os.path.isdir(class_path):
            continue

        output_class_path = os.path.join(OUTPUT_DIR, class_name)
        if not os.path.exists(output_class_path):
            os.makedirs(output_class_path)
            
        print(f"Processando classe: {class_name} e redimensionando para {TARGET_SIZE}x{TARGET_SIZE}...")
        
        for filename in os.listdir(class_path):
            input_file_path = os.path.join(class_path, filename)
            
            is_heic_or_heif = filename.lower().endswith(('.heic', '.heif'))
            is_standard_image = filename.lower().endswith(('.png', '.jpg', '.jpeg'))
            
            if not (is_heic_or_heif or is_standard_image):
                continue
            
            image_rgb = None
            
            try:
                if is_standard_image:
                    image = cv2.imread(input_file_path)
                    if image is not None:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                elif is_heic_or_heif:
                    pi_image = Image.open(input_file_path)
                    image_rgb = np.asarray(pi_image.convert('RGB'))

            except Exception as e:
                print(f"Erro ao ler arquivo {filename}: {e}")
                continue
            
            if image_rgb is None:
                 print(f"Aviso: Não foi possível processar a imagem {filename}")
                 continue

            # --- GERAÇÃO DE MÚLTIPLAS CÓPIAS ---
            base_name = os.path.splitext(filename)[0]
            
            for i in range(N_COPIES):
                augmented_data = pipeline_aumentacao(image=image_rgb)
                augmented_image_rgb = augmented_data['image']
                
                augmented_image_bgr = cv2.cvtColor(augmented_image_rgb, cv2.COLOR_RGB2BGR)
                
                output_filename = f"{base_name}_aug_{i}.jpg"
                output_file_path = os.path.join(output_class_path, output_filename)
                
                # O cv2.imwrite salvará a imagem já redimensionada para 512x512
                cv2.imwrite(output_file_path, augmented_image_bgr)
    
        print(f"Classe {class_name} concluída. {len(os.listdir(output_class_path))} imagens geradas e redimensionadas.")

# 5. Execução do script
if __name__ == '__main__':
    apply_augmentation_batch()
    print(f"\n✅ Processo de Data Augmentation em Lote Finalizado! Todas as imagens estão em {TARGET_SIZE}x{TARGET_SIZE}.")