# 🚀 Ghid Complet de Deployment - Pas cu Pas

## Cuprins
1. [Pre-requisite](#1-pre-requisite)
2. [Antrenare Model](#2-antrenare-model)
3. [Testare Locală](#3-testare-locală)
4. [Docker Build](#4-docker-build)
5. [Deploy pe Fly.io](#5-deploy-pe-flyio)
6. [Troubleshooting](#6-troubleshooting)

---

## 1. Pre-requisite

### Instalări necesare pe PC-ul tău:

```bash
# Python 3.10+
python --version  # verifică versiunea

# pip actualizat
pip install --upgrade pip

# Docker Desktop
# Descarcă de la: https://www.docker.com/products/docker-desktop

# Fly CLI
# Windows (PowerShell):
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Linux/Mac:
curl -L https://fly.io/install.sh | sh

# Verifică instalarea
flyctl version
```

### Structura finală a proiectului:

```
universal-image-classifier/
├── app/
│   └── streamlit_app.py          ✅ Am creat
├── models/
│   ├── image_classifier_model.h5    ⏳ Tu antrenezi
│   └── image_classifier_config.json ⏳ Tu antrenezi
├── sample_images/                   ⏳ Tu adaugi
│   ├── cat_example.jpg
│   ├── dog_example.jpg
│   └── ...
├── universal_classifier.py       ✅ Ai deja
├── Dockerfile                    ✅ Am creat
├── fly.toml                      ✅ Am creat
├── requirements.txt              ✅ Am creat
└── README.md                     ✅ Am creat
```

---

## 2. Antrenare Model

### 2.1 Pregătește dataset-ul

```bash
# Creează structura de foldere
mkdir -p data/animals/{train,validation,test}/{cats,dogs,birds}

# Sau descarcă un dataset gata făcut:
# Kaggle: https://www.kaggle.com/datasets/tongpython/cat-and-dog
```

### 2.2 Rulează antrenarea

```bash
# Navighează în directorul proiectului
cd universal-image-classifier

# Rulează antrenarea
python universal_classifier.py \
    --data_dir data/animals \
    --project_name image_classifier \
    --initial_epochs 10 \
    --finetune_epochs 15

# Durată estimată: 15-30 minute pe GPU, 1-2 ore pe CPU
```

### 2.3 Verifică output-ul

După antrenare ar trebui să ai:
- `image_classifier_model.h5` - modelul antrenat
- `image_classifier_config.json` - clasele și setările
- `image_classifier_training_curves.png` - graficele de antrenare
- `image_classifier_confusion_matrix.png` - matricea de confuzie

### 2.4 Mută fișierele în locul corect

```bash
# Creează folderul models dacă nu există
mkdir -p models

# Mută modelul și config-ul
mv image_classifier_model.h5 models/
mv image_classifier_config.json models/
```

---

## 3. Testare Locală

### 3.1 Instalează dependențele

```bash
# Creează virtual environment
python -m venv venv

# Activează (Windows)
venv\Scripts\activate

# Activează (Linux/Mac)
source venv/bin/activate

# Instalează dependențele
pip install -r requirements.txt
```

### 3.2 Rulează aplicația Streamlit

```bash
# Pornește serverul
streamlit run app/streamlit_app.py

# Se deschide automat în browser la http://localhost:8501
```

### 3.3 Testează funcționalitatea

1. ✅ Upload o imagine
2. ✅ Verifică că primești predicții
3. ✅ Verifică Top-3 predictions
4. ✅ Testează cu imagini diferite

---

## 4. Docker Build

### 4.1 Verifică Docker Desktop

```bash
# Verifică că Docker rulează
docker --version
docker info
```

### 4.2 Build imaginea Docker

```bash
# Build (durează 5-10 minute prima dată)
docker build -t image-classifier:latest .

# Verifică imaginea creată
docker images | grep image-classifier
```

### 4.3 Testează local în Docker

```bash
# Rulează containerul
docker run -p 8080:8080 image-classifier:latest

# Accesează la http://localhost:8080

# Oprește cu Ctrl+C
```

### 4.4 Troubleshooting Docker

```bash
# Dacă build-ul eșuează, verifică spațiul
docker system prune -a

# Rebuild fără cache
docker build --no-cache -t image-classifier:latest .

# Verifică logs-urile
docker logs <container_id>
```

---

## 5. Deploy pe Fly.io

### 5.1 Autentificare

```bash
# Login (se deschide browser-ul)
flyctl auth login

# Verifică autentificarea
flyctl auth whoami
```

### 5.2 Creează aplicația

```bash
# Prima dată - creează aplicația
flyctl launch

# Răspunde la întrebări:
# - App name: universal-image-classifier (sau altul unic)
# - Region: ams (Amsterdam - cel mai aproape de România)
# - PostgreSQL: No
# - Redis: No
```

### 5.3 Deploy

```bash
# Deploy aplicația
flyctl deploy

# Durează 5-15 minute
# Progress: Building → Pushing → Deploying → Success
```

### 5.4 Verifică deployment-ul

```bash
# Status
flyctl status

# Deschide în browser
flyctl open

# Logs în timp real
flyctl logs
```

### 5.5 URL-ul tău

După deploy, aplicația va fi disponibilă la:
```
https://universal-image-classifier.fly.dev
```

---

## 6. Troubleshooting

### Eroare: "Out of Memory"

```bash
# Crește RAM-ul în fly.toml
[[vm]]
  memory_mb = 1024  # sau 2048
```

Apoi:
```bash
flyctl deploy
```

### Eroare: "Model not found"

Verifică că:
1. `models/image_classifier_model.h5` există
2. `models/image_classifier_config.json` există
3. `.dockerignore` nu exclude folderul `models/`

### Eroare: "Container fails to start"

```bash
# Verifică logs
flyctl logs --app universal-image-classifier

# Comun: TensorFlow necesită mai mult RAM
# Soluție: crește memory_mb în fly.toml
```

### Aplicația e lentă

1. Prima încărcare durează mai mult (model loading)
2. Mărește resursele în fly.toml
3. Consideră un model mai mic (MobileNet în loc de VGG16)

---

## 7. Comenzi Utile Fly.io

```bash
# Status aplicație
flyctl status

# Logs
flyctl logs

# SSH în container
flyctl ssh console

# Scale resurse
flyctl scale memory 1024

# Oprește aplicația (economie)
flyctl scale count 0

# Pornește aplicația
flyctl scale count 1

# Ștergere completă
flyctl destroy universal-image-classifier
```

---

## 8. Checklist Final

- [ ] Model antrenat și salvat în `models/`
- [ ] Config JSON în `models/`
- [ ] Imagini sample în `sample_images/`
- [ ] Testat local cu Streamlit
- [ ] Testat local cu Docker
- [ ] Deploy pe Fly.io reușit
- [ ] URL funcțional
- [ ] README actualizat cu URL-ul real
- [ ] Screenshots/demo video făcut

---

## 9. Următorii Pași

După deployment:

1. **Demo Video** - Înregistrează un screencast de 1-2 minute
2. **README** - Actualizează cu URL-ul real și screenshots
3. **LinkedIn Post** - Împărtășește proiectul
4. **CV** - Adaugă link-ul la proiect
5. **Apply** - Trimite la cele 10 joburi planificate!

---

**Succes! 🚀**
