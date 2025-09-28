from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import joblib

from src.utils import train_model, visualize, generate_testcase
from src.loader import load_document 

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Paths
MODEL_PATH = Path("saved_models/trained_model.pkl")
PRETRAINED_PATH = Path("saved_models/pretrained_class_model.pkl")


@app.get("/", response_class=HTMLResponse)
def index():
    """Upload CSV and choose Train/Test or Test Pretrained."""
    return """
    <html><body style="font-family: Arial; margin: 40px;">
      <h2>Upload CSV Dataset</h2>
      <form action="/process/" enctype="multipart/form-data" method="post">
        <label><b>Choose CSV File:</b></label><br>
        <input name="file" type="file" accept=".csv" required><br><br>
        <button type="submit" name="action" value="train">📈 Train Model & Test</button>
        <button type="submit" name="action" value="pretrained">🧪 Test Pretrained Model</button>
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
            plot_path = visualize(temp_path)  # e.g. return "static/plot.png"

            # Generate testcase and test
            testcase = generate_testcase(temp_path)
            prediction = model.predict([testcase])

            return render_result(
                "✅ Training & Testing Complete",
                f"<b>Generated Testcase:</b> {testcase}<br><b>Prediction:</b> {prediction.tolist()}",
                f"<br><img src='/{plot_path}' width='500'>"
            )
        except Exception as e:
            return render_result("❌ Training Failed", str(e))

    # --- TEST PRETRAINED MODEL ---
    elif action == "pretrained":
        try:
            if not PRETRAINED_PATH.exists():
                return render_result("⚠ Pretrained Model Missing",
                                     "Upload pretrained_class_model.pkl under saved_models/.")

            model = joblib.load(PRETRAINED_PATH)

            # Generate testcase and test
            testcase = generate_testcase(temp_path)
            prediction = model.predict([testcase])

            return render_result(
                "🧪 Pretrained Model Testing",
                f"<b>Generated Testcase:</b> {testcase}<br><b>Prediction:</b> {prediction.tolist()}"
            )
        except Exception as e:
            return render_result("❌ Pretrained Testing Failed", str(e))

    return render_result("⚠ Invalid Action", "Please use the provided buttons only.")
