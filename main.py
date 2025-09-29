'''
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from src.utils import train_model, visualize_data, generate_testcase
from src.loader import load_document 

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Paths
MODEL_PATH = Path(r"src\trained_model.pkl")
PRETRAINED_PATH = Path(r"pretrained_class_model.pkl")


@app.get("/", response_class=HTMLResponse)
def index():
    """Upload CSV and choose Train/Test or Test Pretrained."""
    return """
    <html><body style="font-family: Arial; margin: 40px;">
      <h2>Upload CSV Dataset</h2>
      <form action="/process/" enctype="multipart/form-data" method="post">
        <label><b>Choose CSV File:</b></label><br>
        <input name="file" type="file" accept=".csv" required><br><br>
        <button type="submit" name="action" value="train">Train Model & Test</button>
        <button type="submit" name="action" value="pretrained">Test Pretrained Model</button>
      </form>
    </body></html>
    """


@app.post("/process/", response_class=HTMLResponse)
async def process_csv(file: UploadFile = File(...), action: str = Form(...)):
    """Handle training or pretrained testing."""
    # Save uploaded CSV temporarily
    temp_path = Path(f"temp_files/{file.filename}")
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    def render_result(title, message, extra=""):
        return f"""
        <html><body style="font-family: Arial; margin: 40px;">
            <h2>{title}</h2>
            <p>{message}</p>
            {extra}
            <br><br><a href="/">⬅ Go Back</a>
        </body></html>
        """

    # --- TRAIN & TEST ---
    if action == "train":
        try:
            # Train model
            model = train_model(temp_path)
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(model, MODEL_PATH)

            # Visualization (save to static/)
            plot_path = visualize_data(temp_path)  # e.g. return "static/plot.png"

            # Generate testcase and test
            testcase = generate_testcase(temp_path)
            #testcase = np.array(testcase, dtype=float).reshape(1, -1)
            prediction = model.predict([testcase])

            return render_result(
                "Training & Testing Complete",
                f"<b>Generated Testcase:</b> {testcase}<br><b>Prediction:</b> {prediction.tolist()}",
                f"<br><img src='/{plot_path}' width='500'>"
            )
        except Exception as e:
            return render_result("Training Failed", str(e))

    # --- TEST PRETRAINED MODEL ---
    elif action == "pretrained":
        try:
            if not PRETRAINED_PATH.exists():
                return render_result("⚠ Pretrained Model Missing",
                                     "Upload pretrained_class_model.pkl under saved_models/.")

            model = joblib.load(PRETRAINED_PATH)

            # Generate testcase and test
            testcase = generate_testcase(temp_path)
            testcase = pd.DataFrame([testcase])  # wrap dict in a list to make 1-row DataFrame
            non_numeric_cols = testcase.select_dtypes(include=['object']).columns
            testcase = testcase.drop(columns=non_numeric_cols)
            from sklearn.preprocessing import StandardScaler
            num_cols = testcase.select_dtypes(include=['int32', 'int64', 'float32', 'float64']).columns
            scaler = StandardScaler()
            testcase[num_cols] = scaler.fit_transform(testcase[num_cols])
            prediction = model.predict(testcase)

            return render_result(
                "Pretrained Model Testing",
                f"<b>Generated Testcase:</b> {testcase}<br><b>Prediction:</b> {prediction.tolist()}"
            )
        except Exception as e:
            return render_result(" Pretrained Testing Failed", str(e))

    return render_result("⚠ Invalid Action", "Please use the provided buttons only.")
'''

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from pathlib import Path
import joblib
import pandas as pd
from src.utils import train_model, visualize_data, generate_testcase
from src.loader import load_document

app = FastAPI()

# Paths
MODEL_PATH = Path(r"src\trained_model.pkl")
PRETRAINED_PATH = Path(r"pretrained_class_model.pkl")


@app.get("/", response_class=HTMLResponse)
def index():
    """Upload CSV file and choose Train/Test or Test Pretrained. A random testcase wwill be generated in both cases."""
    return """
    <html><body style="font-family: Arial; margin: 40px;">
      <h2>Upload CSV Dataset</h2>
      <form action="/process/" enctype="multipart/form-data" method="post">
        <label><b>Upload CSV File:</b></label><br>
        <input name="file" type="file" accept=".csv" required><br><br>
        <button type="submit" name="action" value="train">Train Model & Test</button>
        <button type="submit" name="action" value="pretrained">Test Pretrained Model</button>
      </form>
    </body></html>
    """


@app.post("/process/", response_class=HTMLResponse)
async def process_csv(file: UploadFile = File(...), action: str = Form(...)):
    """Handle training or pretrained testing."""
    # Save uploaded CSV temporarily
    temp_path = Path(f"temp_files/{file.filename}")
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    def render_result(title, message, df: pd.DataFrame = None, extra=""):
        table_html = ""
        if df is not None:
            table_html = df.to_html(index=False, classes="styled-table", border=0)

        return f"""
        <html>
        <head>
        <style>
            body {{
                font-family: Arial; 
                margin: 40px;
            }}
            .styled-table {{
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 14px;
                min-width: 400px;
                max-width: 90%;
                overflow-x: auto;
                display: block;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .styled-table th, .styled-table td {{
                border: 1px solid #dddddd;
                text-align: center;
                padding: 8px;
            }}
            .styled-table th {{
                background-color: #4CAF50;
                color: white;
            }}
            .styled-table tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .styled-table tr:hover {{
                background-color: #f1f1f1;
            }}
        </style>
        </head>
        <body>
            <h2>{title}</h2>
            <p>{message}</p>
            {table_html}
            {extra}
            <br><br><a href="/">⬅ Go Back</a>
        </body>
        </html>
        """

    # --- TRAIN & TEST ---
    if action == "train":
        try:
            model = train_model(temp_path)
            MODEL_PATH.parent.mkdir(exist_ok=True)
            joblib.dump(model, MODEL_PATH)

            plot_path = visualize_data(temp_path)
            testcase = generate_testcase(temp_path)
            prediction = model.predict([testcase])

            return render_result(
                "Training & Testing Complete",
                f"<b>Prediction:</b> {prediction.tolist()}",
                df=pd.DataFrame([testcase]),
                extra=f"<br><img src='/{plot_path}' width='500'>"
            )
        except Exception as e:
            return render_result("Training Failed", str(e))

    # --- TEST PRETRAINED MODEL ---
    elif action == "pretrained":
        try:
            if not PRETRAINED_PATH.exists():
                return render_result("⚠ Pretrained Model Missing",
                                     "Upload pretrained_class_model.pkl under saved_models/.")

            model = joblib.load(PRETRAINED_PATH)

            testcase = generate_testcase(temp_path)
            testcase = pd.DataFrame([testcase])
            non_numeric_cols = testcase.select_dtypes(include=['object']).columns
            testcase = testcase.drop(columns=non_numeric_cols)

            from sklearn.preprocessing import StandardScaler
            num_cols = testcase.select_dtypes(include=['number']).columns
            #scaler = StandardScaler()
            #testcase[num_cols] = scaler.fit_transform(testcase[num_cols])

            prediction = model.predict(testcase)

            return render_result(
                "Pretrained Model Testing",
                f"<b>Prediction:</b> {prediction.tolist()}",
                df=testcase
            )
        except Exception as e:
            return render_result("Pretrained Testing Failed", str(e))

    return render_result("⚠ Invalid Action", "Please use the provided buttons only.")

