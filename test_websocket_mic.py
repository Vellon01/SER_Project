import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
import warnings

warnings.filterwarnings('ignore')

async def record_and_send(websocket, duration=3.0, sr=22050):
    print(f"\n[Microphone] Recording for {duration} seconds (speak now!)...")
    # Record mono audio as float32
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    print("[Microphone] Recording complete. Sending to WebSocket API...")
    
    # Convert the raw array into WAV formatted bytes in memory
    with io.BytesIO() as wav_io:
        sf.write(wav_io, recording, sr, format='WAV', subtype='PCM_16')
        wav_bytes = wav_io.getvalue()
        
    # Send binary chunk
    await websocket.send(wav_bytes)
    
    # Wait for the API JSON response
    response = await websocket.recv()
    result = json.loads(response)
    
    if "error" in result:
        print(f"Error from server: {result['error']}")
    else:
        print(f">>> Predicted Emotion: {result['emotion'].upper()} (Confidence: {result['confidence']*100:.2f}%)")

async def main():
    uri = "ws://localhost:8000/ws/predict"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Press Ctrl+C to stop.")
            while True:
                input("\nPress [Enter] to start recording a 3-second chunk...")
                await record_and_send(websocket)
    except ConnectionRefusedError:
        print("Could not connect to server. Is the API running? Try running:")
        print("uvicorn api:app --reload")
    except websockets.exceptions.ConnectionClosedError:
        print("Server disconnected.")
    except KeyboardInterrupt:
        print("\nExiting WebSocket Client...")

if __name__ == "__main__":
    asyncio.run(main())
