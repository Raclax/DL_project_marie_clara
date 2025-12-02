import os
from PIL import Image
import shutil
from sklearn.model_selection import train_test_split

def convert_crohme_to_yolo(source_root, dest_root, val_split=0.2):
    """
    Convertit votre dataset CROHME au format YOLO.
    
    source_root : dossier contenant vos PNG + LG
    dest_root : dossier de destination pour le format YOLO
    """
    
    # Créer la structure de dossiers
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dest_root, 'labels', split), exist_ok=True)
    
    # Lister tous les fichiers PNG
    images = [f for f in os.listdir(source_root) if f.endswith('.png')]
    images.sort()
    
    # Split train/val
    train_imgs, val_imgs = train_test_split(images, test_size=val_split, random_state=42)
    
    def map_label(label):
        """Même logique que votre classe"""
        raw = label.split("_")[0].strip()
        if raw.isalpha():
            return 0  # lettres
        if raw.isdigit():
            return 1  # chiffres
        if raw in {"+", "-", "=", "/", "*", "×", "÷", "^"}:
            return 2  # opérateurs
        return 3  # autres
    
    def convert_bbox_to_yolo(xmin, ymin, xmax, ymax, img_width, img_height):
        """Convertit bbox absolue en format YOLO normalisé"""
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height
        return x_center, y_center, width, height
    
    def process_image(img_name, split):
        img_path = os.path.join(source_root, img_name)
        lg_name = img_name.replace('.png', '.lg')
        lg_path = os.path.join(source_root, lg_name)
        
        # Vérifier que le fichier LG existe
        if not os.path.exists(lg_path):
            print(f"Warning: {lg_path} not found, skipping...")
            return
        
        # Obtenir les dimensions de l'image
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Copier l'image
        dest_img_path = os.path.join(dest_root, 'images', split, img_name)
        shutil.copy(img_path, dest_img_path)
        
        # Créer le fichier label YOLO
        yolo_lines = []
        
        with open(lg_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(',') if p.strip()]
                
                if len(parts) < 6:
                    parts = [p.strip() for p in line.strip().split() if p.strip()]
                
                if len(parts) < 6:
                    continue
                
                label = parts[1]
                
                try:
                    xmin = float(parts[-4])
                    ymin = float(parts[-3])
                    xmax = float(parts[-2])
                    ymax = float(parts[-1])
                except Exception:
                    continue
                
                # Vérifier validité de la bbox
                if xmax <= xmin or ymax <= ymin:
                    continue
                
                # Convertir en format YOLO
                class_id = map_label(label)
                x_center, y_center, width, height = convert_bbox_to_yolo(
                    xmin, ymin, xmax, ymax, img_width, img_height
                )
                
                # Vérifier que les valeurs sont dans [0, 1]
                if all(0 <= v <= 1 for v in [x_center, y_center, width, height]):
                    yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        
        # Sauvegarder le fichier label
        label_name = img_name.replace('.png', '.txt')
        label_path = os.path.join(dest_root, 'labels', split, label_name)
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
    
    # Traiter train
    print("Processing train set...")
    for img_name in train_imgs:
        process_image(img_name, 'train')
    
    # Traiter val
    print("Processing val set...")
    for img_name in val_imgs:
        process_image(img_name, 'val')
    
    # Créer le fichier data.yaml
    yaml_content = f"""# CROHME Dataset pour YOLO
path: {os.path.abspath(dest_root)}
train: images/train
val: images/val

# Classes
names:
  0: letter
  1: digit
  2: operator
  3: other

# Nombre de classes
nc: 4
"""
    
    yaml_path = os.path.join(dest_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ Conversion terminée!")
    print(f"   - Train: {len(train_imgs)} images")
    print(f"   - Val: {len(val_imgs)} images")
    print(f"   - Config: {yaml_path}")

# Utilisation
convert_crohme_to_yolo(
    source_root='/home/raclax/Documents/M2/Part2/DL2/Project/datas/FullExpressions/CROHME2019_train_png',
    dest_root='/home/raclax/Documents/M2/Part2/DL2/Project/YOLO_dataset',
    val_split=0.2
)