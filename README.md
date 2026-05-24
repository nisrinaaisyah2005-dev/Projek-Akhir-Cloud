# 🚲 NYC Citi Bike Trips — Dashboard Analisis

Dashboard interaktif berbasis **Streamlit** untuk menganalisis dataset perjalanan NYC Citi Bike.
Data dimuat **langsung dari GitHub** — tidak perlu upload manual.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

## 📁 Struktur Repository

```
adbc_dashboard/
├── app.py                        # Dashboard Streamlit
├── NYC Citi Bike Trips.csv       # Dataset (letakkan di sini)
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml               # Tema warna kuning
```

## ⚙️ Setup Sebelum Deploy

Di `app.py`, temukan baris ini dan **ganti dengan raw URL CSV kamu di GitHub**:

```python
GITHUB_DATA_URL = "https://raw.githubusercontent.com/GITHUB_USERNAME/REPO_NAME/main/NYC%20Citi%20Bike%20Trips.csv"
```

Format URL raw GitHub:
```
https://raw.githubusercontent.com/<username>/<repo>/<branch>/<nama_file.csv>
```

> ⚠️ Pastikan file CSV ada di repository yang **public**, atau gunakan Streamlit Secrets untuk repo private.

## 🚀 Cara Deploy ke Streamlit Cloud

1. Push semua file ke GitHub:
   ```bash
   git init
   git add .
   git commit -m "init dashboard"
   git remote add origin https://github.com/USERNAME/REPO.git
   git push -u origin main
   ```
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Pilih repo, branch `main`, file `app.py`
4. Klik **Deploy**

## 💻 Menjalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📦 Dependencies

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
scikit-learn>=1.3.0
```

## 🎨 Tema

Warna kuning amber — edit `.streamlit/config.toml` untuk menggantinya.
