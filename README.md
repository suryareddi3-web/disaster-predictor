# Disaster Prediction System

This project is a local disaster prediction web application built with Flask and SQLite.

## Features
- Red and white premium user interface
- Two-step location selection: state and district
- Live weather lookup for selected district
- Predict percentage chances for Flood, Wildfire, and Earthquake
- Uses a local cost-free SQLite database and trained model

## Setup
1. Open PowerShell and navigate to the project folder:
   ```powershell
   cd C:\disaster_prediction_mine_clear_2
   ```
2. Create a Python virtual environment if you do not already have one:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
4. Initialize the database and train the prediction model:
   ```powershell
   python init_db.py
   ```
5. Run the app locally:
   ```powershell
   python app.py
   ```
6. Open your browser and visit `http://127.0.0.1:5000` for local testing, or use the deployed URL `https://disaster-predictor.onrender.com`
## Streamlit Deployment

This project now includes a Streamlit app entrypoint at `streamlit_app.py`.

1. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. Run the app locally:
   ```powershell
   streamlit run streamlit_app.py
   ```

For Streamlit Cloud, push this repository to GitHub and connect it to your Streamlit account.
## Docker Deployment

This project is now container-ready with a `Dockerfile` and `.dockerignore`.

Build the image locally:
```powershell
docker build -t disaster-predictor .
```

Run the container:
```powershell
docker run -p 5000:5000 disaster-predictor
```

Then open `http://127.0.0.1:5000` or the public IP/port of your cloud host.
## Notes
- The database is stored in `data/disaster.db`.
- The trained model is stored in `model.pkl`.
- The dataset used for the app is `disaster_dataset.csv`.
- The web form accepts manual weather inputs for the selected state.
- If all weather inputs are left blank, the app attempts to fetch real-time weather from a free API for the selected Indian state.
- If live weather is unavailable, the app falls back to state-average conditions.
- If the app cannot find the database or model file, run `init_db.py` again.
- The app is currently deployed at `https://disaster-predictor.onrender.com`.

## Deployment
This app can be hosted permanently on any Python-compatible web host or PaaS such as Railway, Render, or Heroku.

1. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
2. Set environment variables for production:
   ```powershell
   $env:FLASK_DEBUG = "0"
   $env:FLASK_SECRET_KEY = "your-secret-key"
   $env:APP_PORT = "5000"
   ```
3. Run in production mode:
   ```powershell
   python app.py
   ```

On a cloud host, use the `Procfile` for automatic web startup:
```text
web: waitress-serve --listen=0.0.0.0:$PORT app:app
```

Then open the public URL provided by your host.
