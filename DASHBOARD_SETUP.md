# ECG Real-Time Anomaly Detection Dashboard
## Setup & Installation Guide

---

## 📋 Requirements

Create a `requirements.txt` file:

```
streamlit==1.28.0
numpy==1.24.3
pandas==2.0.3
plotly==5.17.0
torch==2.0.1
scikit-learn==1.3.0
```

---

## 🚀 Installation Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Files

Place these 3 files in the same directory as `ecg_dashboard_app.py`:

- `hp_ecg_final.pt` (your trained model)
- `rr_scaler.pkl` (RR feature scaler)
- `morph_scaler.pkl` (morphological feature scaler)

**From Google Drive:**
```
/content/drive/MyDrive/mit-bih-v2-models/hp_ecg_final.pt
/content/drive/MyDrive/mit-bih-v2-models/rr_scaler.pkl
/content/drive/MyDrive/mit-bih-v2-models/morph_scaler.pkl
```

### 3. Run the Dashboard

```bash
streamlit run ecg_dashboard_app.py
```

The dashboard will open at: `http://localhost:8501`

---

## 👤 Default Login Credentials

**For first-time use, create an account via the "Sign Up" tab.**

Demo accounts you can create:
- Username: `doctor1` | Password: `password123` | Role: Clinician
- Username: `researcher1` | Password: `password123` | Role: Researcher

---

## 🎯 Features

### 1. **Secure Authentication**
- User registration with role-based access (Clinician, Researcher, Administrator)
- Password hashing with SHA-256
- SQLite database for user management
- Session persistence

### 2. **Real-Time ECG Monitoring**
- Live dual-channel ECG streaming (MLII + V1)
- Beat-by-beat anomaly detection
- Real-time probability scoring
- Visual anomaly alerts with red highlighting

### 3. **Alert Timeline**
- Historical view of all detected anomalies
- Timestamp tracking
- Probability distribution over time
- Interactive timeline visualization

### 4. **Performance Analytics**
- Confusion matrix visualization
- ROC curve with AUC score
- Model metrics dashboard
- Per-class performance breakdown

### 5. **Model Information**
- Architecture diagram
- Training configuration details
- Hyperparameters
- Dataset information

---

## 📁 File Structure

```
project/
├── ecg_dashboard_app.py       # Main dashboard application
├── requirements.txt            # Python dependencies
├── hp_ecg_final.pt            # Trained model checkpoint
├── rr_scaler.pkl              # RR feature scaler
├── morph_scaler.pkl           # Morphological feature scaler
└── users.db                   # User database (auto-created)
```

---

## 🔧 Configuration

### Update Model Paths (if needed)

In `ecg_dashboard_app.py`, around line 303:

```python
model_path = "hp_ecg_final.pt"
rr_scaler_path = "rr_scaler.pkl"
morph_scaler_path = "morph_scaler.pkl"
```

Change these to absolute paths if your files are elsewhere:

```python
model_path = "/path/to/your/hp_ecg_final.pt"
rr_scaler_path = "/path/to/your/rr_scaler.pkl"
morph_scaler_path = "/path/to/your/morph_scaler.pkl"
```

### Using Real Test Data

The current dashboard uses **simulated ECG data** for demonstration.

To use real MIT-BIH test data:

1. Load your test set `.npy` files in the `ECGStreamSimulator` class
2. Replace the `generate_beat()` method with actual data loading
3. Update the `get_next_sample()` method to iterate through real beats

Example modification:

```python
class ECGStreamSimulator:
    def __init__(self, test_data_path="X_test.npy", test_labels_path="y_test.npy"):
        self.X_test = np.load(test_data_path)
        self.y_test = np.load(test_labels_path)
        self.current_idx = 0
    
    def get_next_sample(self):
        if self.current_idx >= len(self.X_test):
            self.current_idx = 0  # Loop back
        
        beat = self.X_test[self.current_idx]
        label = self.y_test[self.current_idx]
        self.current_idx += 1
        
        return beat, label
```

---

## 🌐 Deployment Options

### Option 1: Local Development (Recommended for Testing)
```bash
streamlit run ecg_dashboard_app.py
```
Access at: `http://localhost:8501`

### Option 2: Streamlit Cloud (Free Hosting)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo
4. Deploy with one click
5. **Note:** Upload model files to the repo (ensure repo is private if models contain sensitive data)

### Option 3: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "ecg_dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t ecg-dashboard .
docker run -p 8501:8501 ecg-dashboard
```

### Option 4: Cloud VM (AWS, Google Cloud, Azure)

1. Launch a VM instance
2. Install Python 3.10+
3. Upload files via SCP or Git
4. Install dependencies
5. Run with `nohup streamlit run ecg_dashboard_app.py &`
6. Access via VM's public IP: `http://<VM_IP>:8501`

---

## 🔒 Security Notes

### For Production Deployment:

1. **Change Password Hashing:**
   Replace SHA-256 with bcrypt or Argon2:
   ```bash
   pip install bcrypt
   ```
   Update `hash_password()` method.

2. **Add HTTPS:**
   Use reverse proxy (nginx) with SSL certificate.

3. **Database Security:**
   Move from SQLite to PostgreSQL/MySQL for production.

4. **Environment Variables:**
   Store sensitive config in `.env` file:
   ```python
   import os
   from dotenv import load_dotenv
   load_dotenv()
   
   MODEL_PATH = os.getenv("MODEL_PATH")
   ```

5. **Rate Limiting:**
   Add login attempt limits to prevent brute force.

---

## 🐛 Troubleshooting

### Issue: "Model files not found"
**Solution:** Update paths in line 303-305 to absolute paths.

### Issue: "ModuleNotFoundError"
**Solution:** Run `pip install -r requirements.txt` again.

### Issue: "Database is locked"
**Solution:** Close other instances of the app accessing `users.db`.

### Issue: "CUDA out of memory"
**Solution:** Model runs on CPU by default. No GPU needed for inference.

### Issue: "Port 8501 already in use"
**Solution:** 
```bash
streamlit run ecg_dashboard_app.py --server.port=8502
```

---

## 📸 Dashboard Screenshots

After running, you'll see:

1. **Login Page** — Gradient purple background with login/signup tabs
2. **Main Dashboard** — 4 metrics cards + live ECG plot
3. **Alert Timeline** — Interactive timeline of detected anomalies
4. **Analytics Tab** — Confusion matrix + ROC curve
5. **Model Info Tab** — Architecture diagram + config

---

## 📝 Usage Workflow

1. **Sign Up** → Create account with username, password, full name, role
2. **Login** → Enter credentials
3. **Start Monitoring** → Click "▶️ Start Monitoring" button
4. **Watch Live ECG** → See beats stream in real-time with anomaly detection
5. **Check Alerts** → Switch to "Alert Timeline" tab to see history
6. **View Analytics** → Check "Analytics" tab for model performance
7. **Logout** → Click "🚪 Logout" in sidebar

---

## 🎨 Customization

### Change Color Scheme

Update CSS in `st.markdown()` sections around line 427.

### Add More Metrics

Add cards in the metrics row (line 533):
```python
col5, col6 = st.columns(2)
with col5:
    st.metric("Heart Rate", "72 BPM")
with col6:
    st.metric("Avg RR Interval", "833 ms")
```

### Add Audio Alerts

Install `playsound`:
```bash
pip install playsound
```

Add in anomaly detection block:
```python
if pred_anomaly and alert_sound:
    playsound('alert.wav')
```

---

## 🚀 Next Steps

1. **Integrate Real Data:** Replace simulator with actual test set
2. **Add More Arrhythmia Types:** Extend to multi-class classification
3. **Export Reports:** Add PDF report generation
4. **Mobile App:** Build Flutter/React Native wrapper
5. **Cloud Sync:** Integrate with hospital EHR systems

---

## 📞 Support

For issues or questions, refer to:
- Streamlit docs: https://docs.streamlit.io
- PyTorch docs: https://pytorch.org/docs
- Plotly docs: https://plotly.com/python

---

END OF SETUP GUIDE
