// Tab Switching Logic
function setDemoMode(mode) {
    // If we navigate away from the webcam, explicitly tell the hardware to stop
    if (mode !== 'webcam') {
        stopWebcam();
    }

    // Hide all viewers
    document.querySelectorAll('.viewer').forEach(el => el.classList.add('hidden'));

    // Remove active class from buttons
    document.querySelectorAll('.method-card').forEach(el => el.classList.remove('active'));

    // Show selected viewer
    document.getElementById(`${mode}-view`).classList.remove('hidden');
    document.getElementById(`btn-${mode}`).classList.add('active');

    // Smooth scroll to demo section
    document.getElementById('lab').scrollIntoView({ behavior: 'smooth' });
}

// Webcam Logic
const webcamImg = document.getElementById('webcam-stream');
const webcamPlaceholder = document.getElementById('webcam-placeholder');
const stopWebcamBtn = document.getElementById('stop-webcam-btn');
const demoDisplay = document.querySelector('.demo-display');
const hiddenCanvas = document.getElementById('webcam-canvas');
const ctx = hiddenCanvas.getContext('2d', { willReadFrequently: true });

let ws = null;
let mediaStream = null;
let frameInterval = null;
let isStreaming = false;

async function startWebcam() {
    try {
        // 1. Request webcam access from the browser
        mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });

        // Feed the native stream into a hidden video element to draw from
        const hiddenVideo = document.createElement('video');
        hiddenVideo.srcObject = mediaStream;
        await hiddenVideo.play();

        // 2. Open WebSocket connection
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws/webcam`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            isStreaming = true;
            hiddenCanvas.width = hiddenVideo.videoWidth;
            hiddenCanvas.height = hiddenVideo.videoHeight;

            // Loop: Capture frame -> Send to server over WS
            frameInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN && isStreaming) {
                    ctx.drawImage(hiddenVideo, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
                    const base64Frame = hiddenCanvas.toDataURL('image/jpeg', 0.8);
                    ws.send(base64Frame);
                }
            }, 100); // 10 FPS
        };

        // 3. Receive processed frames from server and display them
        ws.onmessage = (event) => {
            webcamImg.src = event.data;
        };

        // UI Updates
        webcamPlaceholder.classList.add('hidden');
        webcamImg.classList.remove('hidden');
        stopWebcamBtn.classList.remove('hidden');
        demoDisplay.classList.add('glow');

    } catch (err) {
        console.error("Camera access denied or error:", err);
        alert("Failed to access camera. Please ensure permissions are granted.");
    }
}

function stopWebcam() {
    isStreaming = false;

    // Stop capturing frames
    if (frameInterval) clearInterval(frameInterval);

    // Close WebSocket
    if (ws) {
        ws.close();
        ws = null;
    }

    // Shutdown hardware camera
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }

    // UI Updates
    webcamImg.src = "";
    webcamImg.classList.add('hidden');
    stopWebcamBtn.classList.add('hidden');
    webcamPlaceholder.classList.remove('hidden');
    demoDisplay.classList.remove('glow');
}

// Image Upload Logic
document.getElementById('image-input').addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show loading UI
    document.getElementById('image-upload-box').classList.add('hidden');
    document.getElementById('image-loading').classList.remove('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch('/detect-image', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);

            // Show result
            document.getElementById('image-loading').classList.add('hidden');
            document.getElementById('image-result').classList.remove('hidden');
            document.getElementById('processed-image').src = imageUrl;
            demoDisplay.classList.add('glow');
        } else {
            alert('Error processing image');
            resetImage();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error during image processing');
        resetImage();
    }
});

function resetImage() {
    document.getElementById('image-input').value = "";
    document.getElementById('image-result').classList.add('hidden');
    document.getElementById('image-upload-box').classList.remove('hidden');
    demoDisplay.classList.remove('glow');
}

// Video Upload Logic
document.getElementById('video-input').addEventListener('change', async function (e) {
    const file = e.target.files[0];
    if (!file) return;

    // Show loading UI
    document.getElementById('video-upload-box').classList.add('hidden');
    document.getElementById('video-loading').classList.remove('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch('/detect-video', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const blob = await response.blob();
            const videoUrl = URL.createObjectURL(blob);

            // Show result
            document.getElementById('video-loading').classList.add('hidden');
            document.getElementById('video-result').classList.remove('hidden');

            const videoEl = document.getElementById('processed-video');
            videoEl.src = videoUrl;

            // Setup download button fallback
            const downloadBtn = document.getElementById('video-download-btn');
            downloadBtn.href = videoUrl;
            demoDisplay.classList.add('glow');

        } else {
            alert('Error processing video: ' + (await response.text()));
            resetVideo();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Network error during video processing');
        resetVideo();
    }
});

function resetVideo() {
    document.getElementById('video-input').value = "";
    document.getElementById('video-result').classList.add('hidden');
    document.getElementById('video-upload-box').classList.remove('hidden');
    demoDisplay.classList.remove('glow');
}
