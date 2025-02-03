import os
import cv2
import numpy as np

def load_images_with_labels(base_dir):
    """
    디렉토리에서 이미지를 로드하고 라벨을 생성하여 numpy 배열로 반환합니다.
    """
    images = []  # 이미지 데이터를 저장할 리스트
    labels = []  # 라벨 정보를 저장할 리스트

    # 디렉토리 탐색
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일 필터링
                file_path = os.path.join(root, file)
                try:
                    # 이미지 로드 (흑백)
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f"[WARN] Skipping invalid image: {file_path}")
                        continue

                    # 이미지 리사이즈 (233x233)
                    img_resized = cv2.resize(img, (233, 233))
                    images.append(img_resized)

                    # 라벨 생성: 'HCM' 또는 'NORMAL'
                    if "HCM" in file_path:
                        labels.append(1)  # HCM은 라벨 1
                    elif "NORMAL" in file_path:
                        labels.append(0)  # NORMAL은 라벨 0

                    print(f"[INFO] Loaded image: {file_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to process image {file_path}: {e}")

    # numpy 배열로 변환
    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int32)

    return images_np, labels_np

def save_numpy_arrays(images, labels, output_dir, prefix):
    """
    이미지와 라벨 배열을 numpy 파일로 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # numpy 파일 저장
    images_path = os.path.join(output_dir, f"{prefix}_images.npy")
    labels_path = os.path.join(output_dir, f"{prefix}_labels.npy")
    
    np.save(images_path, images)
    np.save(labels_path, labels)
    
    print(f"[INFO] Saved images to: {images_path}")
    print(f"[INFO] Saved labels to: {labels_path}")

if __name__ == "__main__":
    base_dir = r"C:\Users\user\Desktop\팀프로젝트\Cat_HCM\Data\CLAHE_processed_images\CLAHE_train_validation_images"
    output_dir = r"C:\Users\user\Desktop\팀프로젝트\Cat_HCM\Data\npy_files"

    print("[INFO] Processing train dataset...")
    train_images, train_labels = load_images_with_labels(os.path.join(base_dir, "HCM", "train"))
    save_numpy_arrays(train_images, train_labels, output_dir, "train_HCM")

    print("[INFO] Processing val dataset...")
    val_images, val_labels = load_images_with_labels(os.path.join(base_dir, "HCM", "val"))
    save_numpy_arrays(val_images, val_labels, output_dir, "val_HCM")

    print("[INFO] Processing NORMAL train dataset...")
    normal_train_images, normal_train_labels = load_images_with_labels(os.path.join(base_dir, "NORMAL", "train"))
    save_numpy_arrays(normal_train_images, normal_train_labels, output_dir, "train_NORMAL")

    print("[INFO] Processing NORMAL val dataset...")
    normal_val_images, normal_val_labels = load_images_with_labels(os.path.join(base_dir, "NORMAL", "val"))
    save_numpy_arrays(normal_val_images, normal_val_labels, output_dir, "val_NORMAL")

    print("[INFO] All datasets processed.")
