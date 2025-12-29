# Signature Verification System - Backend API

A robust, high-performance Django REST Framework backend for offline signature verification. This system leverages a hybrid approach combining **Deep Learning (CNNs, Siamese Networks)** and **Handcrafted Features (Geometric, Texture, Structural)** to achieve high accuracy in distinguishing genuine signatures from forgeries.

## 🚀 System Architecture

```mermaid
graph TD
    Client[Frontend Client] -->|REST API| API_Gateway[Django REST API]
    API_Gateway -->|Auth| Auth_Module[Authentication System]
    API_Gateway -->|Image Upload| Preprocessor[Image Preprocessing Pipeline]
    
    subgraph Core Verification Engine
        Preprocessor -->|Cleaned Image| FE_CNN[VGG16 Feature Extractor]
        Preprocessor -->|Cleaned Image| FE_Geo[Geometric Analysis]
        Preprocessor -->|Cleaned Image| FE_Tex[Texture (GLCM/LBP)]
        Preprocessor -->|Cleaned Image| FE_Str[Structural (Skeleton)]
        
        FE_CNN -->|Vector| Hybrid_Model[Weighted Hybrid Model]
        FE_Geo -->|Vector| Hybrid_Model
        FE_Tex -->|Vector| Hybrid_Model
        FE_Str -->|Vector| Hybrid_Model
        
        Hybrid_Model -->|Similarity Score| Decision_Logic[Threshold Logic]
    end
    
    Decision_Logic -->|JSON Result| API_Gateway
    API_Gateway -->|Store Result| DB[(MySQL Database)]
```

## 🧠 Technical Details

The verification core (`integrated_sig_verifier.py`) utilizes a weighted ensemble of algorithms:

### 1. Deep Learning Features (CNN) - *Weight: 65%*
-   **Model**: Pre-trained **VGG16** (ImageNet weights).
-   **Layer**: `block5_pool` output.
-   **Purpose**: Extracts high-level visual patterns and abstract features robust to minor variations.

### 2. Similarity Learning (Siamese) - *Weight: 25%*
-   **Architecture**: Custom Siamese Network optimized for signature pairs.
-   **Loss Function**: Contrastive Loss.
-   **Metric**: Euclidean Distance between feature vectors.

### 3. Handcrafted Features (10%)
-   **Geometric (1%)**: Analyzes aspect ratios, mass distribution (centroids), and stroke envelope.
-   **Texture (2%)**: Uses **GLCM** (Gray-Level Co-occurrence Matrix) and **LBP** (Local Binary Patterns) to detect ink pressure variances.
-   **Structural (1%)**: Skeletonizes the signature to analyze stroke width, junctions, and endpoints.
-   **Sequence (6%)**: LSTM-based analysis of stroke sequences (simulated from static images).

## 🛠 Tech Stack

-   **Framework**: Django 5.2, Django REST Framework
-   **ML/CV**: TensorFlow 2.10, OpenCV 4.7, scikit-image, scikit-learn
-   **Database**: MySQL
-   **Authentication**: Token-based Auth (DRF)

## 🔌 API Endpoints

### Authentication
-   `POST /api/auth/login/` - Obtain auth token.
-   `POST /api/auth/register/` - Register new user.

### Signatures
-   `GET /api/signatures/` - List verified signatures.
-   `POST /api/signatures/` - Upload new reference signature.

### Verification
-   `POST /api/verify/`
    -   **Payload**: `{"test_signature": <image_file>, "user_profile_id": <id>}`
    -   **Response**:
        ```json
        {
            "result": "genuine",
            "confidence": 98.5,
            "metrics": { "structure_score": 0.92, "texture_score": 0.88 }
        }
        ```

## ⚡ Installation & Setup

### Prerequisites
-   Python 3.9+
-   MySQL Server
-   CUDA Toolkit (optional, for GPU acceleration)

### Steps

1.  **Clone Repository**
    ```bash
    git clone https://github.com/your-repo/signature-verification-backend.git
    cd signature-verification-backend
    ```

2.  **Environment Setup**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configuration**
    Create a `.env` file in the root directory:
    ```ini
    SECRET_KEY=your_secret_key
    DB_PASSWORD=your_mysql_password
    ```

4.  **Database Migration**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

5.  **Run Server**
    ```bash
    python manage.py runserver
    ```

## ⚖️ License
MIT License
