const recordBtn = document.getElementById('recordButton');
const statusText = document.getElementById('statusText');
const emotionDisplay = document.getElementById('emotionDisplay');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceFill = document.getElementById('confidenceFill');

let isRecording = false;
let audioContext;
let processor;
let mediaStream;
let ws;
let audioBuffer = [];
const CHUNK_TIME_MS = 3000;
let chunkTimer = null;
let currentSampleRate = 22050; 

// Emotion to color map
const colorMap = {
    'neutral': 'var(--color-neutral)',
    'calm': 'var(--color-calm)',
    'happy': 'var(--color-happy)',
    'sad': 'var(--color-sad)',
    'angry': 'var(--color-angry)',
    'fearful': 'var(--color-fearful)',
    'disgusted': 'var(--color-disgusted)',
    'surprised': 'var(--color-surprised)',
    'unknown': '#ffffff'
};

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Encodes the raw PCM data directly into a WAV array buffer
function encodeWAV(samples, sampleRate) {
    let buffer = new ArrayBuffer(44 + samples.length * 2);
    let view = new DataView(buffer);
    
    // RIFF chunk descriptor
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    
    // FMT sub-chunk
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true); 
    view.setUint16(20, 1, true); 
    view.setUint16(22, 1, true); 
    view.setUint32(24, sampleRate, true); 
    view.setUint32(28, sampleRate * 2, true); 
    view.setUint16(32, 2, true); 
    view.setUint16(34, 16, true); 
    
    // Data sub-chunk
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    // Write PCM data
    floatTo16BitPCM(view, 44, samples);
    
    return buffer;
}


function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws/predict`);
    
    ws.onopen = () => {
        console.log("WebSocket connected");
        statusText.innerText = "Connected. Click to Start.";
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if(data.error) {
            console.error("Server API Error:", data.error);
            // Ignore small silence errors, show text otherwise
            if(!data.error.includes("too short")) {
                statusText.innerText = "Error: See Console";
            }
            return;
        }
        
        displayResult(data.emotion, data.confidence);
    };
    
    ws.onclose = () => {
        console.log("WebSocket disconnected. Reconnecting in 3s...");
        statusText.innerText = "Disconnected.";
        setTimeout(connectWebSocket, 3000);
        if(isRecording) stopRecording();
    };
}

function displayResult(emotion, confidence) {
    emotionDisplay.classList.remove('hidden');
    confidenceBar.classList.remove('hidden');
    emotionDisplay.classList.add('visible');
    confidenceBar.classList.add('visible');
    
    const root = document.documentElement;
    const color = colorMap[emotion.toLowerCase()] || '#ffffff';
    root.style.setProperty('--active-color', color);
    
    emotionDisplay.innerText = emotion;
    emotionDisplay.style.color = color;
    
    confidenceFill.style.width = `${confidence * 100}%`;
    confidenceFill.style.backgroundColor = color;
}

async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Use Web Audio API
        audioContext = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 22050});
        currentSampleRate = audioContext.sampleRate;
        
        const source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        processor.onaudioprocess = (e) => {
            if(!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            audioBuffer.push(new Float32Array(inputData));
        };
        
        isRecording = true;
        recordBtn.classList.add('recording');
        statusText.innerText = "Listening... Speak now";
        
        // Start chunk interval
        chunkTimer = setInterval(sendAudioChunk, CHUNK_TIME_MS);
        
    } catch (err) {
        console.error("Microphone access denied:", err);
        statusText.innerText = "Mic Access Denied. Check permissions.";
    }
}

function sendAudioChunk() {
    if (audioBuffer.length === 0 || ws.readyState !== WebSocket.OPEN) return;
    
    // Concatenate bits
    let totalLength = audioBuffer.reduce((acc, val) => acc + val.length, 0);
    let combined = new Float32Array(totalLength);
    let offset = 0;
    for(let i=0; i < audioBuffer.length; i++) {
        combined.set(audioBuffer[i], offset);
        offset += audioBuffer[i].length;
    }
    
    audioBuffer = [];
    
    // Encode down to WAV buffer so Python can instantly read it
    const wavBuffer = encodeWAV(combined, currentSampleRate);
    ws.send(wavBuffer);
}

function stopRecording() {
    if (processor) processor.disconnect();
    if (audioContext) audioContext.close();
    if (mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    
    clearInterval(chunkTimer);
    isRecording = false;
    recordBtn.classList.remove('recording');
    statusText.innerText = "Click to Start Recording";
    audioBuffer = [];
}

recordBtn.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
});

// Initialize WebSocket immediately
connectWebSocket();
