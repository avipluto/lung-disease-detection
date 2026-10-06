# 🫁 LungScan AI - Lung Disease Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1-orange?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker" alt="Docker">
</p>

An AI-powered chest X-ray analysis system that detects **14 thoracic diseases** using a **DenseNet121 (CheXNet)** deep learning model. Built with FastAPI backend, modern web frontend, and Docker containerization.

---

## ✨ Features

- 🧠 **DenseNet121 Architecture** — State-of-the-art CheXNet-inspired model
- 🏥 **14 Disease Classes** — Comprehensive thoracic disease detection
- ⚡ **FastAPI Backend** — High-performance async API
- 🎨 **Modern Web UI** — Drag & drop upload with dark/light mode
- 🐳 **Docker Ready** — One-command deployment with docker-compose
- 📊 **Confidence Scores** — Color-coded prediction bars with medical descriptions
- 🔒 **Privacy First** — All processing happens locally

## 🏗️ Architecture

```
┌─────────────────┐     HTTP      ┌──────────────────────┐
│                 │    /api/*     │                      │
│   Frontend      │◄────────────►│   FastAPI Backend     │
│   (Nginx)       │              │                      │
│   Port 3000     │              │   Port 8000          │
│                 │              │                      │
│  - HTML/CSS/JS  │              │  - REST API          │
│  - Drag & Drop  │              │  - Image Processing  │
│  - Dark Mode    │              │  - DenseNet121 Model │
└─────────────────┘              └──────────────────────┘
```

## 🔬 Supported Diseases

| # | Disease | Description |
|---|---------|-------------|
| 1 | Atelectasis | Partial/complete lung collapse |
| 2 | Cardiomegaly | Enlarged heart |
| 3 | Effusion | Fluid in pleural space |
| 4 | Infiltration | Abnormal substance in lung tissue |
| 5 | Mass | Opacity >3cm (possible cancer) |
| 6 | Nodule | Small rounded opacity <3cm |
| 7 | Pneumonia | Lung infection/inflammation |
| 8 | Pneumothorax | Collapsed lung |
| 9 | Consolidation | Fluid-filled lung region |
| 10 | Edema | Excess fluid in lungs |
| 11 | Emphysema | Damaged air sacs |
| 12 | Fibrosis | Lung tissue scarring |
| 13 | Pleural Thickening | Thickened pleural lining |
| 14 | Hernia | Organ protrusion through diaphragm |

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/lung-disease-detection.git
cd lung-disease-detection

# Build and run with Docker Compose
docker-compose up --build

# Open in browser
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
# Simply open frontend/index.html in your browser
# Or serve with any HTTP server:
cd frontend
python -m http.server 3000
```

## 🏋️ Training Your Own Model

1. **Download the NIH ChestX-ray14 dataset** from [NIH Box](https://nihcc.app.box.com/v/ChestXray-NIHCC)

2. **Organize data:**
   ```
   data/
   ├── images/              # All X-ray images
   └── Data_Entry_2017.csv  # Labels file
   ```

3. **Install training dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   pip install pandas scikit-learn tqdm tensorboard
   ```

4. **Run training:**
   ```bash
   python train.py \
       --data_dir data/ \
       --epochs 20 \
       --batch_size 32 \
       --lr 0.0001 \
       --output_dir backend/models
   ```

5. **Monitor with TensorBoard:**
   ```bash
   tensorboard --logdir backend/models/runs
   ```

The trained model will be saved to `backend/models/model.pth` and automatically used by the API.

## 📡 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API welcome message |
| `GET` | `/api/health` | Health check & model status |
| `POST` | `/api/predict` | Analyze chest X-ray image |
| `GET` | `/docs` | Interactive API documentation |

### Example API Usage

```python
import requests

# Upload and predict
with open("chest_xray.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/predict",
        files={"file": f}
    )

results = response.json()
for pred in results["predictions"]:
    print(f"{pred['disease']}: {pred['confidence']:.1f}%")
```

## 📁 Project Structure

```
lung-disease-detection/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI application
│   │   ├── model.py         # DenseNet121 model
│   │   └── schemas.py       # Pydantic schemas
│   ├── models/              # Model weights (.pth)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── index.html           # Main UI
│   ├── style.css            # Styles
│   ├── script.js            # Frontend logic
│   ├── nginx.conf           # Nginx configuration
│   └── Dockerfile
├── docker-compose.yml        # Container orchestration
├── train.py                  # Model training script
├── README.md
├── LICENSE
├── .gitignore
└── .dockerignore
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **ML Model** | PyTorch, DenseNet121, torchvision |
| **Backend** | FastAPI, Uvicorn, Python 3.11 |
| **Frontend** | HTML5, CSS3, JavaScript (Vanilla) |
| **Containerization** | Docker, Docker Compose, Nginx |
| **Training** | PyTorch, scikit-learn, TensorBoard |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

> **This tool is for educational and research purposes only.** It is NOT a substitute for professional medical diagnosis, treatment, or advice. Always consult a qualified healthcare provider for any medical concerns. The predictions made by this AI model should not be used as the sole basis for any clinical decision.

---

<p align="center">
  Built with ❤️ by Aviral Yadav using PyTorch & FastAPI
</p>
