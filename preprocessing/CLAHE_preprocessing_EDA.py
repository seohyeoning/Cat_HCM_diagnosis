import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model

#############################
# 1) 폴더 경로 설정
#############################
base_dir = "C:/Users/user/Desktop/팀프로젝트/Cat_HCM/Data/classified_train/train and validation images"
hcm_dir = os.path.join(base_dir, "HCM")
normal_dir = os.path.join(base_dir, "NORMAL")

#############################
# 2) CLAHE 적용 함수 정의
#############################
def apply_clahe(gray_img, clip_limit=2.0, tile_grid_size=(8,8)):
    """
    CLAHE를 적용하여 이미지의 대비를 향상시킵니다.
    입력: 흑백(Grayscale) 이미지 (0~1 범위)
    출력: CLAHE 적용 후 이미지 (0~1 범위)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    # 0~1 범위를 0~255로 스케일링
    gray_img_uint8 = (gray_img * 255).astype(np.uint8)
    # CLAHE 적용
    clahe_img = clahe.apply(gray_img_uint8)
    # 다시 0~1 범위로 정규화
    clahe_img_normalized = clahe_img / 255.0
    return clahe_img_normalized

#############################
# 3) 이미지 로드 및 CLAHE 적용 함수 정의
#############################
def load_and_process_images(folder_path, label, apply_clahe_flag=False):
    """
    이미지를 로드하고, 전처리(리사이즈, 정규화, CLAHE 적용)를 수행합니다.
    입력:
        - folder_path: 이미지가 저장된 폴더 경로
        - label: 이미지의 라벨 (HCM=1, Normal=0)
        - apply_clahe_flag: CLAHE 적용 여부 (True/False)
    출력:
        - images: 전처리된 이미지 리스트
        - labels: 라벨 리스트
    """
    images = []
    labels = []
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            fpath = os.path.join(folder_path, fname)
            img_bgr = cv2.imread(fpath)
            if img_bgr is not None:
                # Grayscale 변환
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                # 리사이즈
                img_resized = cv2.resize(img_gray, (233, 233))
                # 정규화
                img_normalized = img_resized / 255.0
                # CLAHE 적용 여부
                if apply_clahe_flag:
                    img_processed = apply_clahe(img_normalized)
                else:
                    img_processed = img_normalized
                # 리스트에 추가
                images.append(img_processed)
                labels.append(label)
    return images, labels

#############################
# 4) 데이터 불러오기
#############################
print("[INFO] Loading and processing images without CLAHE...")
X_hcm, y_hcm = load_and_process_images(hcm_dir, label=1, apply_clahe_flag=False)
X_normal, y_normal = load_and_process_images(normal_dir, label=0, apply_clahe_flag=False)

X_list = X_hcm + X_normal
y_list = y_hcm + y_normal

# NumPy 배열로 변환
X_original = np.array(X_list, dtype=np.float32)  # shape: (N, 233, 233)
y_original = np.array(y_list, dtype=np.int32)    # shape: (N,)

print("[INFO] Original data loaded:", X_original.shape, y_original.shape)

#############################
# 5) CLAHE 적용하여 데이터 불러오기
#############################
print("[INFO] Loading and processing images with CLAHE...")
X_hcm_clahe, y_hcm_clahe = load_and_process_images(hcm_dir, label=1, apply_clahe_flag=True)
X_normal_clahe, y_normal_clahe = load_and_process_images(normal_dir, label=0, apply_clahe_flag=True)

X_clahe_list = X_hcm_clahe + X_normal_clahe
y_clahe_list = y_hcm_clahe + y_normal_clahe

# NumPy 배열로 변환
X_clahe = np.array(X_clahe_list, dtype=np.float32)  # shape: (N, 233, 233)
y_clahe = np.array(y_clahe_list, dtype=np.int32)    # shape: (N,)

print("[INFO] CLAHE processed data loaded:", X_clahe.shape, y_clahe.shape)

#############################
# 6) Train/Val Split
#############################
print("[INFO] Splitting original data into Train/Val...")
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(
    X_original, y_original,
    test_size=0.2,
    stratify=y_original,
    random_state=42
)

print("[INFO] Original Train shape:", X_train_orig.shape, y_train_orig.shape)
print("[INFO] Original Validation shape:", X_val_orig.shape, y_val_orig.shape)

print("[INFO] Splitting CLAHE data into Train/Val...")
X_train_clahe, X_val_clahe, y_train_clahe, y_val_clahe = train_test_split(
    X_clahe, y_clahe,
    test_size=0.2,
    stratify=y_clahe,
    random_state=42
)

print("[INFO] CLAHE Train shape:", X_train_clahe.shape, y_train_clahe.shape)
print("[INFO] CLAHE Validation shape:", X_val_clahe.shape, y_val_clahe.shape)

#############################
# 7) 라벨 분포 확인
#############################
def print_label_distribution(y, dataset_name="Dataset"):
    unique_labels, counts = np.unique(y, return_counts=True)
    label_dict = {0: 'Normal', 1: 'HCM'}
    label_counts = {label_dict[k]: v for k, v in zip(unique_labels, counts)}
    print(f"[EDA] {dataset_name} 라벨 분포:", label_counts)

print_label_distribution(y_original, "전체 데이터")
print_label_distribution(y_train_orig, "Original Train")
print_label_distribution(y_val_orig, "Original Validation")

print_label_distribution(y_clahe, "CLAHE 전체 데이터")
print_label_distribution(y_train_clahe, "CLAHE Train")
print_label_distribution(y_val_clahe, "CLAHE Validation")

#############################
# 8) 픽셀 통계 확인
#############################
def print_pixel_statistics(X, dataset_name="Dataset"):
    mean = X.mean()
    std = X.std()
    print(f"{dataset_name} 픽셀 평균: {mean:.4f}")
    print(f"{dataset_name} 픽셀 표준편차: {std:.4f}")

print_pixel_statistics(X_train_orig, "Original Train")
print_pixel_statistics(X_val_orig, "Original Validation")

print_pixel_statistics(X_train_clahe, "CLAHE Train")
print_pixel_statistics(X_val_clahe, "CLAHE Validation")

#############################
# 9) 전처리 전/후 샘플 이미지 시각화
#############################
def visualize_clahe_effect(X_orig, X_clahe, y, num_samples=3):
    """
    CLAHE 적용 전후의 샘플 이미지를 시각화합니다.
    """
    random_indices = np.random.choice(len(X_orig), size=num_samples, replace=False)
    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3*num_samples))
    
    for i, idx in enumerate(random_indices):
        original_img = X_orig[idx]
        clahe_img = X_clahe[idx]
        label = y[idx]
        
        # Original Image
        axes[i, 0].imshow(original_img, cmap='gray')
        axes[i, 0].set_title(f"Original (Label={label})")
        axes[i, 0].axis('off')
        
        # CLAHE Image
        axes[i, 1].imshow(clahe_img, cmap='gray')
        axes[i, 1].set_title(f"CLAHE (Label={label})")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig("clahe_effect_comparison.png")  # 이미지 저장
    plt.show()

print("[INFO] Visualizing CLAHE effect on Train data...")
visualize_clahe_effect(X_train_orig, X_train_clahe, y_train_orig, num_samples=3)

#############################
# 10) PCA + K-means 시각화
#############################
def pca_kmeans_visualization(X, y, dataset_name="Dataset"):
    """
    PCA를 통해 2D로 축소 후 K-means 군집화와 실제 라벨을 시각화합니다.
    """
    # Flatten
    X_flat = X.reshape(len(X), -1)
    
    # PCA 50차원
    pca_50 = PCA(n_components=50, random_state=42)
    X_pca_50 = pca_50.fit_transform(X_flat)
    
    # PCA 2차원
    pca_2 = PCA(n_components=2, random_state=42)
    X_pca_2d = pca_2.fit_transform(X_pca_50)
    
    # K-means
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_pca_50)
    
    # 시각화
    plt.figure(figsize=(16,6))
    
    plt.subplot(1,2,1)
    plt.scatter(X_pca_2d[:,0], X_pca_2d[:,1], c=clusters, cmap='viridis', alpha=0.6)
    plt.title(f"K-means clusters (2D PCA) - {dataset_name}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.subplot(1,2,2)
    plt.scatter(X_pca_2d[:,0], X_pca_2d[:,1], c=y, cmap='coolwarm', alpha=0.6)
    plt.title(f"Actual labels (2D PCA) - {dataset_name}")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    
    plt.tight_layout()
    plt.savefig(f"PCA_KMeans_{dataset_name}.png")  # 이미지 저장
    plt.show()

print("[INFO] PCA + K-means visualization for Original Train...")
pca_kmeans_visualization(X_train_orig, y_train_orig, "Original Train")

print("[INFO] PCA + K-means visualization for CLAHE Train...")
pca_kmeans_visualization(X_train_clahe, y_train_clahe, "CLAHE Train")

#############################
# 11) t-SNE 시각화
#############################
def tsne_visualization(X, y, dataset_name="Dataset", sample_size=1000):
    """
    t-SNE를 이용하여 고차원 데이터를 2D로 시각화합니다.
    """
    if len(X) > sample_size:
        idxs = np.random.choice(len(X), size=sample_size, replace=False)
        X_sample = X[idxs]
        y_sample = y[idxs]
    else:
        X_sample = X
        y_sample = y
    
    # Flatten
    X_flat = X_sample.reshape(len(X_sample), -1)
    
    # PCA 50차원
    pca_50 = PCA(n_components=50, random_state=42)
    X_pca_50 = pca_50.fit_transform(X_flat)
    
    # t-SNE 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne_2d = tsne.fit_transform(X_pca_50)
    
    # 시각화
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(X_tsne_2d[:,0], X_tsne_2d[:,1], c=y_sample, cmap='coolwarm', alpha=0.6)
    plt.title(f"t-SNE Visualization - {dataset_name}")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.colorbar(scatter, label="Label (0=Normal, 1=HCM)")
    plt.savefig(f"tSNE_{dataset_name}.png")  # 이미지 저장
    plt.show()

print("[INFO] t-SNE visualization for Original Train...")
tsne_visualization(X_train_orig, y_train_orig, "Original Train")

print("[INFO] t-SNE visualization for CLAHE Train...")
tsne_visualization(X_train_clahe, y_train_clahe, "CLAHE Train")

print("=== CLAHE Preprocessing & EDA Completed ===")

#############################
# 12) (C) CLAHE 적용 후 사전학습 모델 Embedding EDA
#############################
print("\n[C] CLAHE 적용 후 사전학습 모델 Embedding EDA")

# 12.1. ResNet50 모델 로드 (최상위 분류층 제외)
print("[INFO] Loading ResNet50 model for embedding extraction...")
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(233, 233, 3))
# Global Average Pooling
embedding_model = Model(inputs=base_model.input, outputs=base_model.output)
embedding_model = Model(inputs=base_model.input, outputs=base_model.output)
print("[INFO] ResNet50 model loaded.")

# 12.2. 임베딩 추출 함수 정의
def extract_embeddings(X, model, batch_size=32):
    """
    주어진 모델을 사용하여 이미지 데이터의 임베딩을 추출합니다.
    입력:
        - X: 이미지 데이터 (N, H, W)
        - model: 사전학습된 모델
        - batch_size: 배치 크기
    출력:
        - embeddings: 추출된 임베딩 (N, Features)
    """
    # Reshape X to (N, H, W, C) and duplicate channels if necessary
    X_reshaped = np.stack([X]*3, axis=-1)  # Convert grayscale to RGB by duplicating channels
    embeddings = model.predict(preprocess_input(X_reshaped), batch_size=batch_size, verbose=1)
    # Flatten embeddings
    embeddings_flat = embeddings.reshape(embeddings.shape[0], -1)
    return embeddings_flat

# 12.3. Original Train 데이터 임베딩 추출
print("[INFO] Extracting embeddings for Original Train data...")
embeddings_orig_train = extract_embeddings(X_train_orig, embedding_model)

# 12.4. CLAHE Train 데이터 임베딩 추출
print("[INFO] Extracting embeddings for CLAHE Train data...")
embeddings_clahe_train = extract_embeddings(X_train_clahe, embedding_model)

# 12.5. PCA + t-SNE 시각화 함수 정의
def embedding_visualization(embeddings_orig, embeddings_clahe, y, dataset_name="Dataset"):
    """
    원본과 CLAHE 적용된 임베딩 데이터를 비교 시각화합니다.
    """
    # PCA 50D
    pca_50 = PCA(n_components=50, random_state=42)
    embeddings_pca_orig = pca_50.fit_transform(embeddings_orig)
    embeddings_pca_clahe = pca_50.transform(embeddings_clahe)
    
    # t-SNE 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_tsne_orig = tsne.fit_transform(embeddings_pca_orig)
    embeddings_tsne_clahe = tsne.fit_transform(embeddings_pca_clahe)
    
    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original 임베딩
    scatter1 = axes[0].scatter(embeddings_tsne_orig[:,0], embeddings_tsne_orig[:,1], c=y, cmap='coolwarm', alpha=0.6)
    axes[0].set_title(f"t-SNE Embeddings - Original Train ({dataset_name})")
    axes[0].set_xlabel("t-SNE Component 1")
    axes[0].set_ylabel("t-SNE Component 2")
    
    # CLAHE 임베딩
    scatter2 = axes[1].scatter(embeddings_tsne_clahe[:,0], embeddings_tsne_clahe[:,1], c=y, cmap='coolwarm', alpha=0.6)
    axes[1].set_title(f"t-SNE Embeddings - CLAHE Train ({dataset_name})")
    axes[1].set_xlabel("t-SNE Component 1")
    axes[1].set_ylabel("t-SNE Component 2")
    
    plt.colorbar(scatter2, ax=axes, label="Label (0=Normal, 1=HCM)")
    plt.tight_layout()
    plt.savefig(f"Embedding_tSNE_{dataset_name}.png")  # 이미지 저장
    plt.show()

# 12.6. 임베딩 시각화
print("[INFO] Visualizing embeddings with t-SNE...")
embedding_visualization(embeddings_orig_train, embeddings_clahe_train, y_train_orig, "CLAHE Applied vs Original")

print("=== (C) CLAHE 적용 후 사전학습 모델 Embedding EDA Completed ===")
