from flask import Flask, render_template, request
import json
import subprocess
import sys


app = Flask(
    __name__,
    template_folder="../frontend",
    static_folder="../frontend",
    static_url_path="/static"
)


# Dashboard
@app.route("/")
def home():

    with open("employees.json", "r") as file:
        employees = json.load(file)

    return render_template(
        "index.html",
        employees=employees
    )


# Add Employee
@app.route("/add_employee", methods=["POST"])
def add_employee():

    employee_id = request.form["employee_id"]
    employee_name = request.form["employee_name"]

    subprocess.Popen([
        sys.executable,
        "capture_faces.py",
        employee_id,
        employee_name
    ])

    return """
    <h2>Employee Added Successfully!</h2>

    <p>Face capture has started.</p>

    <p>
        After capturing the face images,
        return to the dashboard and train the model.
    </p>

    <br>

    <a href="/">Back to Dashboard</a>
    """


# Train Model
@app.route("/train")
def train():

    subprocess.run([
        sys.executable,
        "blur_augmentation.py"
    ])

    subprocess.run([
        sys.executable,
        "train_blur_model.py"
    ])

    return """
    <h2>Blur-Robust LBPH Model Trained Successfully!</h2>

    <p>
        All registered employees with training images are included.
    </p>

    <br>

    <a href="/">Back to Dashboard</a>
    """


# Face Recognition
@app.route("/recognition")
def recognition():

    subprocess.Popen([
        sys.executable,
        "recognize.py"
    ])

    return """
    <h2>Face Recognition Started</h2>

    <p>Check the camera window.</p>

    <br>

    <a href="/">Back to Dashboard</a>
    """


# Attendance
@app.route("/attendance", methods=["POST"])
def attendance():

    session_minutes = request.form["session_minutes"]

    required_percentage = request.form["required_percentage"]

    subprocess.Popen([
        sys.executable,
        "attendance.py",
        session_minutes,
        required_percentage
    ])

    return """
    <h2>Attendance System Started</h2>

    <p>
        The attendance camera has started.
    </p>

    <p>
        Check the camera window for attendance tracking.
    </p>

    <br>

    <a href="/">Back to Dashboard</a>
    """


# Results
@app.route("/results")
def results():

    return render_template(
        "results.html"
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )