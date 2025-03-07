from flask import Flask, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/insights')
def insights():
    # Run the detector_gui.py Streamlit app on port 8501
    subprocess.Popen(['streamlit', 'run', 'detector_gui.py', '--server.port=8505'])
    return redirect("http://localhost:8505")

@app.route('/detector')
def filter():
    # Run the image_filter.py Streamlit app on port 8502
    subprocess.Popen(['streamlit', 'run', 'image_filter.py', '--server.port=8513'])
    return redirect("http://localhost:8513")

if __name__ == '__main__':
    app.run(debug=True, port=8970)