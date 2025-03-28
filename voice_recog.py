from flask import Flask, jsonify, render_template_string, request
import subprocess
import os
import time
import numpy as np
import librosa
import sqlite3
import soundfile as sf
from scipy.spatial.distance import cosine
from scipy.signal import butter, filtfilt
import tempfile
from speechbrain.pretrained import SpeakerRecognition

app = Flask(__name__)

# Recording settings
RECORDING_DIR = "/home/nakul/Downloads"
STREAM_URL = "http://192.0.0.2:8080/audio.wav"  # Your phone's IP
DURATION = 10  # seconds
SAMPLE_RATE = 16000
DB_FILE = "speaker_database.db"

# Create recordings directory if it doesn't exist
os.makedirs(RECORDING_DIR, exist_ok=True)

# Initialize SpeechBrain model
spk_rec = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                          savedir="tmp_model")

def initialize_db():
    """Initialize SQLite database for storing speaker embeddings."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS speakers 
                      (name TEXT PRIMARY KEY, embedding BLOB)''')
    conn.commit()
    conn.close()

 

def register_speaker(name, audio_file):
    """Registers a speaker by storing their voice embedding in the database."""
    try:
        embedding = extract_embedding(audio_file)
        if embedding is None or embedding.size == 0:
            return f"Failed to extract voice features for {name}. Please try again."
            
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO speakers (name, embedding) VALUES (?, ?)",
                       (name, embedding.tobytes()))
        conn.commit()
        conn.close()
        return f"Speaker '{name}' registered successfully."
    except Exception as e:
        raise Exception(f"Error in register_speaker: {str(e)}")

def identify_speaker(audio_file):
    """Identifies the speaker by comparing the input voice sample to registered embeddings."""
    try:
        input_embedding = extract_embedding(audio_file)
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT name, embedding FROM speakers")
        speakers = cursor.fetchall()
        conn.close()
        
        if not speakers:
            return {
                "status": "error",
                "message": "No registered speakers found.",
                "probabilities": {}
            }

        best_match = None
        best_score = float("inf")
        all_scores = {}

        for name, emb_blob in speakers:
            stored_embedding = np.frombuffer(emb_blob, dtype=np.float32)
            # Ensure the dimensions match before calculating cosine distance
            if stored_embedding.shape != input_embedding.shape:
                continue
                
            score = cosine(input_embedding, stored_embedding)
            all_scores[name] = f"{(1-score)*100:.2f}%"
            if score < best_score:
                best_score = score
                best_match = name

        # If no valid comparisons were made
        if best_match is None:
            return {
                "status": "error",
                "message": "Could not compare voice with registered users.",
                "probabilities": {}
            }

        confidence = (1 - best_score) * 100
        
        # Threshold for speaker similarity
        if best_score < 0.7:  
            return {
                "status": "granted",
                "message": f"Welcome {best_match}",
                "person": best_match,
                "confidence": f"{confidence:.2f}%",
                "probabilities": all_scores
            }
        else:
            return {
                "status": "denied",
                "message": "ACCESS DENIED",
                "confidence": f"{confidence:.2f}%",
                "probabilities": all_scores
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error during identification: {str(e)}",
            "probabilities": {}
        }

def butter_highpass(cutoff, fs, order=5):
    """Design a highpass filter"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_highpass_filter(data, cutoff, fs, order=5):
    """Apply highpass filter to remove low-frequency noise"""
    b, a = butter_highpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)

def remove_noise(audio, sr):
    """Apply noise reduction techniques"""
    # 1. High-pass filter (removes low frequency noise, e.g., humming)
    audio_filtered = apply_highpass_filter(audio, cutoff=100, fs=sr)
    
    # 2. Simple noise gate (silence parts with low amplitude)
    noise_gate_threshold = 0.01
    audio_gated = audio_filtered.copy()
    audio_gated[np.abs(audio_gated) < noise_gate_threshold] = 0
    
    # 3. Spectral subtraction (estimate noise from first 0.5s and subtract)
    if len(audio) > sr * 0.5:  # Ensure we have at least 0.5s of audio
        noise_sample = audio[:int(sr * 0.5)]
        noise_spectrum = np.mean(np.abs(librosa.stft(noise_sample))**2, axis=1)
        audio_stft = librosa.stft(audio_gated)
        audio_spec = np.abs(audio_stft)**2
        
        # Calculate reduction factor (avoiding complete elimination)
        reduction_factor = 0.7
        
        # Apply spectral subtraction with flooring to avoid negative values
        audio_spec_sub = np.maximum(audio_spec - reduction_factor * noise_spectrum.reshape(-1, 1), 0.01 * audio_spec)
        
        # Convert back to time domain
        audio_denoised = librosa.istft(audio_spec_sub**(1/2) * np.exp(1j * np.angle(audio_stft)))
        
        # Ensure same length as original
        if len(audio_denoised) >= len(audio):
            audio_denoised = audio_denoised[:len(audio)]
        else:
            audio_denoised = np.pad(audio_denoised, (0, len(audio) - len(audio_denoised)))
            
        return audio_denoised
    
    return audio_gated

@app.route('/')
def index():
    # Web page with voice recognition and registration functionality
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Recognition</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial; text-align: center; padding: 20px; }
            .record-btn { 
                background-color: #f44336; 
                border: none; 
                color: white; 
                padding: 40px; 
                text-align: center; 
                border-radius: 50%; 
                font-size: 24px; 
                cursor: pointer; 
                margin: 30px; 
            }
            .register-btn {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
            .status { margin-top: 20px; font-size: 18px; }
            .result { 
                margin-top: 30px; 
                padding: 20px; 
                border-radius: 10px; 
                font-size: 24px; 
                font-weight: bold;
            }
            .granted { background-color: #4CAF50; color: white; }
            .denied { background-color: #f44336; color: white; }
            .error { background-color: #ff9800; color: white; }
            .confidence { font-size: 16px; margin-top: 10px; }
            .details { 
                margin-top: 20px; 
                text-align: left; 
                max-width: 400px; 
                margin-left: auto; 
                margin-right: auto;
                font-size: 14px;
            }
            .tab {
                overflow: hidden;
                border: 1px solid #ccc;
                background-color: #f1f1f1;
                border-radius: 5px 5px 0 0;
            }
            .tab button {
                background-color: inherit;
                float: left;
                border: none;
                outline: none;
                cursor: pointer;
                padding: 14px 16px;
                transition: 0.3s;
                font-size: 17px;
            }
            .tab button:hover {
                background-color: #ddd;
            }
            .tab button.active {
                background-color: #ccc;
            }
            .tabcontent {
                display: none;
                padding: 6px 12px;
                border: 1px solid #ccc;
                border-top: none;
                border-radius: 0 0 5px 5px;
                animation: fadeEffect 1s;
                margin-bottom: 20px;
            }
            .form-group {
                margin: 15px 0;
                text-align: left;
            }
            .form-group label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
            }
            .form-group input {
                padding: 10px;
                width: 90%;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-size: 16px;
            }
            @keyframes fadeEffect {
                from {opacity: 0;}
                to {opacity: 1;}
            }
        </style>
    </head>
    <body>
        <h1>Voice Recognition System</h1>
        
        <div class="tab">
            <button class="tablinks active" onclick="openTab(event, 'Verify')">Verify Voice</button>
            <button class="tablinks" onclick="openTab(event, 'Register')">Register New User</button>
        </div>
        
        <div id="Verify" class="tabcontent" style="display: block;">
            <h2>Verify Your Voice</h2>
            <button class="record-btn" id="verifyBtn" onclick="recordAudio('verify')">RECORD</button>
            <div class="status" id="verifyStatus">Press the button to start recording</div>
            <div id="verifyResultContainer" style="display: none;">
                <div class="result" id="verifyResult"></div>
                <div class="confidence" id="verifyConfidence"></div>
                <div class="details" id="verifyDetails"></div>
            </div>
        </div>
        
        <div id="Register" class="tabcontent">
            <h2>Register New User</h2>
            <div class="form-group">
                <label for="userName">User Name:</label>
                <input type="text" id="userName" placeholder="Enter your name">
            </div>
            <button class="register-btn" id="registerBtn" onclick="registerUser()">RECORD & REGISTER</button>
            <div class="status" id="registerStatus">Fill in your name and press the button to start recording</div>
            <div id="registerResultContainer" style="display: none;">
                <div class="result" id="registerResult"></div>
            </div>
        </div>
        
        <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
            function recordAudio(mode) {
                const btn = mode === 'verify' ? 'verifyBtn' : 'registerBtn';
                const status = mode === 'verify' ? 'verifyStatus' : 'registerStatus';
                const resultContainer = mode === 'verify' ? 'verifyResultContainer' : 'registerResultContainer';
                
                document.getElementById(btn).disabled = true;
                document.getElementById(status).innerText = 'Recording started...';
                document.getElementById(resultContainer).style.display = 'none';
                
                let endpoint = '/record';
                if (mode === 'register') {
                    const userName = document.getElementById('userName').value;
                    if (!userName) {
                        document.getElementById(status).innerText = 'Please enter a name first!';
                        document.getElementById(btn).disabled = false;
                        return;
                    }
                    endpoint = `/record?userName=${encodeURIComponent(userName)}`;
                }
                
                fetch(endpoint)
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById(status).innerText = data.message;
                        
                        // Start countdown
                        let seconds = ''' + str(DURATION) + ''';
                        const countdown = setInterval(() => {
                            document.getElementById(status).innerText = 
                                `Recording in progress: ${seconds} seconds remaining...`;
                            seconds--;
                            
                            if (seconds < 0) {
                                clearInterval(countdown);
                                document.getElementById(status).innerText = 'Processing recording...';
                                
                                // Check result after recording is done
                                if (mode === 'verify') {
                                    setTimeout(checkResult, 1000, data.file_path);
                                } else {
                                    setTimeout(checkRegistration, 1000, data.file_path, data.user_name);
                                }
                            }
                        }, 1000);
                    });
            }
            
            function checkResult(filePath) {
                fetch('/classify?file=' + filePath)
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('verifyStatus').innerText = 'Analysis complete';
                        document.getElementById('verifyResultContainer').style.display = 'block';
                        
                        const resultElem = document.getElementById('verifyResult');
                        resultElem.innerText = data.message;
                        resultElem.className = 'result ' + data.status;
                        
                        if (data.confidence) {
                            document.getElementById('verifyConfidence').innerText = 'Confidence: ' + data.confidence;
                        } else {
                            document.getElementById('verifyConfidence').innerText = '';
                        }
                        
                        // Show detailed probabilities
                        let detailsHtml = '<h3>Detailed Results:</h3>';
                        if (Object.keys(data.probabilities).length > 0) {
                            for (const [person, prob] of Object.entries(data.probabilities)) {
                                detailsHtml += `<div>${person}: ${prob}</div>`;
                            }
                        } else {
                            detailsHtml += '<div>No matching data available</div>';
                        }
                        document.getElementById('verifyDetails').innerHTML = detailsHtml;
                        
                        // Re-enable record button
                        document.getElementById('verifyBtn').disabled = false;
                    });
            }
            
            function registerUser() {
                recordAudio('register');
            }
            
            function checkRegistration(filePath, userName) {
                fetch(`/register?file=${filePath}&name=${encodeURIComponent(userName)}`)
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('registerStatus').innerText = 'Registration complete';
                        document.getElementById('registerResultContainer').style.display = 'block';
                        
                        const resultElem = document.getElementById('registerResult');
                        resultElem.innerText = data.message;
                        resultElem.className = data.status === 'success' ? 'result granted' : 'result error';
                        
                        // Re-enable register button
                        document.getElementById('registerBtn').disabled = false;
                    });
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/record')
def record():
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    user_name = request.args.get('userName', '')
    
    if user_name:
        output_path = f"{RECORDING_DIR}/register-{user_name}-{timestamp}.wav"
    else:
        output_path = f"{RECORDING_DIR}/verify-{timestamp}.wav"
    
    try:
        # Use subprocess to call ffmpeg
        subprocess.Popen([
            'ffmpeg',
            '-i', STREAM_URL,
            '-t', str(DURATION),
            '-acodec', 'copy',
            '-y',
            output_path
        ])
        
        return jsonify({
            "status": "success", 
            "message": f"Recording for {DURATION} seconds...",
            "file_path": output_path,
            "user_name": user_name
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Recording error: {str(e)}"
        })

@app.route('/register')
def register():
    file_path = request.args.get('file')
    name = request.args.get('name')
    
    if not name:
        return jsonify({
            "status": "error",
            "message": "No name provided for registration"
        })
    
    # Wait for file to be completely written
    max_wait = 5  # Maximum wait time in seconds
    wait_time = 0
    while not os.path.exists(file_path) and wait_time < max_wait:
        time.sleep(0.5)
        wait_time += 0.5
    
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": "Recording file not found"
        })
        
    # Give ffmpeg a moment to finish writing the file
    time.sleep(2)
    
    try:
        # Register the speaker
        message = register_speaker(name, file_path)
        return jsonify({
            "status": "success",
            "message": message
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error during registration: {str(e)}"
        })

@app.route('/classify')
def classify():
    file_path = request.args.get('file')
    
    if not file_path:
        return jsonify({
            "status": "error",
            "message": "No file path provided",
            "probabilities": {}
        })
    
    # Wait for file to be completely written
    max_wait = 5  # Maximum wait time in seconds
    wait_time = 0
    while not os.path.exists(file_path) and wait_time < max_wait:
        time.sleep(0.5)
        wait_time += 0.5
    
    if not os.path.exists(file_path):
        return jsonify({
            "status": "error",
            "message": "Recording file not found",
            "probabilities": {}
        })
        
    # Give ffmpeg a moment to finish writing the file
    time.sleep(2)
    
    # Identify the speaker
    result = identify_speaker(file_path)
    return jsonify(result)

if __name__ == '__main__':
    # Initialize the database
    initialize_db()
    app.run(host='0.0.0.0', port=5000, debug=True)