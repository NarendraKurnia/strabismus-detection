// ============================================================
// STRABISMUS DETECTION — Camera, Bounding Box & Detection Logic
// ============================================================

// --- State ---
let videoStream = null;
let animationFrameId = null;
const videoEl = document.getElementById('videoElement');
const overlayCanvas = document.getElementById('overlayCanvas');
const captureCanvas = document.getElementById('captureCanvas');
const placeholder = document.getElementById('cameraPlaceholder');

// --- Bounding Box Config (50% lebar × 70% tinggi, centered) ---
const BOX_RATIO = { wPercent: 0.50, hPercent: 0.70 };

// ============================================================
// 1. Tab Switching
// ============================================================
function switchTab(tab) {
    // Deactivate all tabs
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(sec => sec.classList.remove('active'));

    if (tab === 'upload') {
        document.getElementById('tabUpload').classList.add('active');
        document.getElementById('uploadSection').classList.add('active');
        stopCamera(); // Matikan kamera saat pindah tab
    } else {
        document.getElementById('tabCamera').classList.add('active');
        document.getElementById('cameraSection').classList.add('active');
    }
}

// ============================================================
// 2. Camera Stream
// ============================================================
async function startCamera() {
    try {
        videoStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 } },
            audio: false
        });

        videoEl.srcObject = videoStream;
        videoEl.classList.add('active');
        placeholder.classList.add('hidden');

        document.getElementById('btnStartCamera').style.display = 'none';
        document.getElementById('btnCapture').disabled = false;
        document.getElementById('btnStopCamera').style.display = '';

        // Mulai render overlay setelah video siap
        videoEl.addEventListener('loadedmetadata', () => {
            overlayCanvas.width = videoEl.videoWidth;
            overlayCanvas.height = videoEl.videoHeight;
            drawOverlay();
        }, { once: true });

    } catch (err) {
        alert('❌ Gagal mengakses kamera: ' + err.message + '\n\nPastikan Anda mengizinkan akses kamera di browser.');
    }
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }

    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    videoEl.srcObject = null;
    videoEl.classList.remove('active');
    placeholder.classList.remove('hidden');

    document.getElementById('btnStartCamera').style.display = '';
    document.getElementById('btnCapture').disabled = true;
    document.getElementById('btnStopCamera').style.display = 'none';
}

// ============================================================
// 3. Bounding Box Overlay (Real-time Canvas)
// ============================================================
function getBoundingBoxRect(canvasW, canvasH) {
    const boxW = Math.floor(canvasW * BOX_RATIO.wPercent);
    const boxH = Math.floor(canvasH * BOX_RATIO.hPercent);
    const boxX = Math.floor((canvasW - boxW) / 2);
    const boxY = Math.floor((canvasH - boxH) / 2);
    return { x: boxX, y: boxY, w: boxW, h: boxH };
}

function drawOverlay() {
    const ctx = overlayCanvas.getContext('2d');
    const cw = overlayCanvas.width;
    const ch = overlayCanvas.height;

    ctx.clearRect(0, 0, cw, ch);

    const box = getBoundingBoxRect(cw, ch);

    // 1. Dark overlay di luar bounding box
    ctx.fillStyle = 'rgba(0, 0, 0, 0.35)';
    // Top
    ctx.fillRect(0, 0, cw, box.y);
    // Bottom
    ctx.fillRect(0, box.y + box.h, cw, ch - box.y - box.h);
    // Left
    ctx.fillRect(0, box.y, box.x, box.h);
    // Right
    ctx.fillRect(box.x + box.w, box.y, cw - box.x - box.w, box.h);

    // 2. Dashed border bounding box
    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 3;
    ctx.setLineDash([12, 6]);
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.setLineDash([]);

    // 3. Corner accents (tebal, solid)
    const cornerLen = 25;
    ctx.strokeStyle = '#00FFFF';
    ctx.lineWidth = 5;
    ctx.lineCap = 'round';

    // Top-left
    drawCorner(ctx, box.x, box.y, cornerLen, 1, 1);
    // Top-right
    drawCorner(ctx, box.x + box.w, box.y, cornerLen, -1, 1);
    // Bottom-left
    drawCorner(ctx, box.x, box.y + box.h, cornerLen, 1, -1);
    // Bottom-right
    drawCorner(ctx, box.x + box.w, box.y + box.h, cornerLen, -1, -1);

    // 4. Label "AREA WAJAH"
    ctx.font = '700 14px Poppins, sans-serif';
    ctx.fillStyle = 'rgba(0, 255, 255, 0.9)';
    ctx.shadowColor = 'rgba(0, 0, 0, 0.7)';
    ctx.shadowBlur = 4;
    ctx.fillText('AREA WAJAH', box.x + 12, box.y + 24);
    ctx.shadowBlur = 0;

    // 5. Hint text (bottom)
    const hintText = 'Posisikan wajah Anda di dalam kotak ini';
    ctx.font = '500 12px Poppins, sans-serif';
    const hintMetrics = ctx.measureText(hintText);
    const hintX = (cw - hintMetrics.width) / 2;
    const hintY = box.y + box.h + 28;

    if (hintY < ch - 10) {
        // Background pill
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        const pillPad = 12;
        const pillH = 24;
        ctx.beginPath();
        roundRect(ctx, hintX - pillPad, hintY - 16, hintMetrics.width + pillPad * 2, pillH, 12);
        ctx.fill();

        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(hintText, hintX, hintY);
    }

    // Loop animation frame
    animationFrameId = requestAnimationFrame(drawOverlay);
}

function drawCorner(ctx, x, y, len, dirX, dirY) {
    ctx.beginPath();
    ctx.moveTo(x, y + len * dirY);
    ctx.lineTo(x, y);
    ctx.lineTo(x + len * dirX, y);
    ctx.stroke();
}

function roundRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
}

// ============================================================
// 4. Capture & Detect (Live Camera)
// ============================================================
async function captureAndDetect() {
    if (!videoStream) return;

    const vw = videoEl.videoWidth;
    const vh = videoEl.videoHeight;

    // Capture frame dari video
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, vw, vh);

    // Encode sebagai base64
    const dataURL = captureCanvas.toDataURL('image/jpeg', 0.92);

    // Hitung ROI (sama dengan overlay)
    const roi = getBoundingBoxRect(vw, vh);

    // Kirim ke server
    showLoading(true);
    try {
        const response = await fetch('/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: dataURL, roi: roi, mode: 'camera' })
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || 'Server error');
        }

        const result = await response.json();
        displayCameraResults(result);
    } catch (err) {
        alert('❌ Gagal mendeteksi: ' + err.message);
    } finally {
        showLoading(false);
    }
}

function displayCameraResults(result) {
    const container = document.getElementById('cameraResults');
    container.style.display = 'block';

    document.getElementById('cameraPreview').src = result.preview_image;
    document.getElementById('cameraResult').src = result.result_image;

    renderDiagnosa('cameraDiagnosa', result.status);

    // Scroll ke hasil
    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// 5. Upload & Detect
// ============================================================
const uploadDropZone = document.getElementById('uploadDropZone');
const fileInput = document.getElementById('fileInput');

if (uploadDropZone) {
    uploadDropZone.addEventListener('click', () => fileInput.click());

    uploadDropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadDropZone.classList.add('dragover');
    });

    uploadDropZone.addEventListener('dragleave', () => {
        uploadDropZone.classList.remove('dragover');
    });

    uploadDropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadDropZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) handleUploadFile(files[0]);
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) handleUploadFile(e.target.files[0]);
    });
}

async function handleUploadFile(file) {
    // Validasi tipe
    const allowed = ['image/jpeg', 'image/png', 'image/bmp'];
    if (!allowed.includes(file.type)) {
        alert('❌ Format file tidak didukung. Gunakan JPG, PNG, atau BMP.');
        return;
    }

    // Baca sebagai base64
    const reader = new FileReader();
    reader.onload = async function (ev) {
        const dataURL = ev.target.result;

        // Tampilkan preview asli
        document.getElementById('uploadOriginal').src = dataURL;

        // Kirim ke server (tanpa ROI — full image)
        showLoading(true);
        try {
            const response = await fetch('/detect', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL, mode: 'upload' })
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error');
            }

            const result = await response.json();
            displayUploadResults(result);
        } catch (err) {
            alert('❌ Gagal mendeteksi: ' + err.message);
        } finally {
            showLoading(false);
        }
    };
    reader.readAsDataURL(file);
}

function displayUploadResults(result) {
    const container = document.getElementById('uploadResults');
    container.style.display = 'block';

    document.getElementById('uploadResult').src = result.result_image;

    renderDiagnosa('uploadDiagnosa', result.status);

    container.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================
// 6. Diagnosa Renderer
// ============================================================
function renderDiagnosa(containerId, status) {
    const container = document.getElementById(containerId);
    let html = '<h3 style="margin-bottom: 12px; font-size: 18px;">📊 Hasil Analisis</h3>';

    if (status === 'STRABISMUS (JULING)') {
        html += `
            <div class="diagnosa-card diagnosa-error">
                <div class="diagnosa-title">🚨 DIAGNOSA AWAL: ${status}</div>
            </div>
            <div class="diagnosa-card diagnosa-warning">
                ⚠️ <strong>Rekomendasi:</strong> Deteksi cermin AI mendeteksi adanya indikasi sudut juling pada mata. 
                Disarankan untuk melakukan pemeriksaan bersama Dokter Spesialis Mata.
            </div>
        `;
    } else if (status === 'TIDAK TERDETEKSI') {
        html += `
            <div class="diagnosa-card diagnosa-warning">
                <div class="diagnosa-title">🔍 STATUS: ${status}</div>
            </div>
            <div class="diagnosa-card diagnosa-info">
                Sistem kesulitan menemukan posisi mata Anda dengan jelas. 
                Pastikan wajah tegak lurus menghadap kamera tanpa terhalang rambut atau kacamata.
            </div>
        `;
    } else {
        html += `
            <div class="diagnosa-card diagnosa-success">
                <div class="diagnosa-title">✅ DIAGNOSA AWAL: ${status} (Kondisi Normal)</div>
            </div>
            <div class="diagnosa-card diagnosa-info">
                👁️ Sumbu bola mata kanan dan kiri Anda sejajar serta simetris.
            </div>
        `;
    }

    container.innerHTML = html;
}

// ============================================================
// 7. Loading Overlay
// ============================================================
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}
