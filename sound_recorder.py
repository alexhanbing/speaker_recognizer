import pyaudio
import wave

class SoundRecorder(object):
    def __init__(self, length_of_audio=5, format=pyaudio.paInt16,
        channels=2, rate=44100, chunk=1024):
        self.length_of_audio = length_of_audio
        self.format = format
        self.channels = channels
        self.rate = rate
        self.chunk = chunk

    def record_audio(self, audio_file_name, length_of_audio):
        audio = pyaudio.PyAudio()

        # start recording
        stream = audio.open(format=self.format, channels=self.channels,
            rate=self.rate, input=True, frames_per_buffer=self.chunk)
        print 'recording...'
        frames = []

        for i in range(0, int(self.rate / self.chunk * length_of_audio)):
            data = stream.read(self.chunk)
            frames.append(data)
        print 'finished recording'

        # finish recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(audio_file_name, 'wb')
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(audio.get_sample_size(self.format))
        wave_file.setframerate(self.rate)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
