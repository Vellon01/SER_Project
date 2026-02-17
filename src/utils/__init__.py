class Reader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read(self):
        with open(self.file_path, 'r') as f:
            return f.read()
    
    def read_wav(self):
        import wave
        with wave.open(self.file_path, 'rb') as wav_file:
            return wav_file.readframes(wav_file.getnframes())
    
    #make a func for all the audio files

    def read_audio(self):
        pass
