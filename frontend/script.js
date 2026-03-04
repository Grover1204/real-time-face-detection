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

function startWebcam() {
    webcamPlaceholder.classList.add('hidden');
    webcamImg.classList.remove('hidden');
    stopWebcamBtn.classList.remove('hidden');

    // Add glowing border to the container
    demoDisplay.classList.add('glow');

    // Add cache-busting query param to fix browser caching of the MJPEG stream
    webcamImg.src = "/webcam-stream?" + new Date().getTime();
}

async function stopWebcam() {
    webcamImg.src = "";
    webcamImg.classList.add('hidden');
    stopWebcamBtn.classList.add('hidden');
    webcamPlaceholder.classList.remove('hidden');
    demoDisplay.classList.remove('glow');

    // Explicitly notify backend to kill the hardware feed
    try {
        await fetch('/stop-webcam', { method: 'POST' });
    } catch (e) {
        console.error("Failed to cleanly disconnect the webcam", e);
    }
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
