# YOLO Detection Training Platform

Platform web untuk training model deteksi objek berbasis YOLO (YOLOv8) dengan fitur lengkap: CRUD project, upload dataset, anotasi bounding box, training, dan inferensi.

## Fitur

1. **Dashboard + CRUD Project** - Kelola project deteksi
2. **Upload Dataset** - Upload gambar ke project
3. **Anotasi** - Tandai objek dengan bounding box, simpan langsung ke project
4. **Training** - Pilih model YOLO (yolov8n/s/m/l/x), training di web, download best.pt
5. **Inferensi** - Deteksi dengan input: gambar, video, kamera webcam, atau URL/RTSP

## Instalasi

```bash
cd AI_Detection_trainer
python -m venv venv
source venv/bin/activate  # Linux/Mac
# atau: venv\Scripts\activate  # Windows

pip install -r requirements.txt
python app.py
```

Buka http://localhost:5000

## Struktur

```
├── app.py              # Entry point
├── config.py           # Konfigurasi
├── models.py           # Model database
├── routes/             # Blueprint routes
├── templates/          # HTML templates
├── static/             # JS, CSS
└── data/               # Upload, projects, models (auto-created)
```

## Menjalankan di Google Colab / Kaggle

1. Push project ke GitHub
2. Buka `colab_run.ipynb` di Colab atau Kaggle
3. Edit `GIT_REPO_URL` dengan URL repo Anda
4. Jalankan semua cell berurutan
5. Klik link yang muncul untuk akses platform

## Alur Kerja

1. Buat project baru
2. Klik project → Upload dataset gambar
3. Klik tiap gambar → Anotasi dengan bounding box → Simpan
4. Setelah ada anotasi → Training → Pilih model YOLO → Mulai
5. Setelah selesai → Download best.pt
6. Inferensi: upload gambar/video, atau gunakan kamera/RTSP
