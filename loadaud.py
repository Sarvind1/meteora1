import numpy as np
import spectrogram as S
def get_audio(a):
    input_signal = S.open_wavfile(a)
    #label_signal = S.open_wavfile('Data/{}.wav'.format(b))
    spectrogram = S.audio_to_spectrogram(input_signal, 512, 160, 400)
    #label_spectrogram = S.audio_to_spectrogram(label_signal, 512, 160, 400)
    spectrogram = np.transpose(spectrogram)
    #label_spectrogram = np.transpose(label_spectrogram)
    y = spectrogram.shape[0]
    #z = label_spectrogram.shape[0]
    if y>298:
        spectrogram = spectrogram[:298-y,:]
    real_part = spectrogram.real
    imag_part = spectrogram.imag
    split_channel = np.array([[real_part, imag_part]])
    #print (split_channel.shape,"#print1")
    split_channel = split_channel.reshape([1,298,257,2])
    #print (split_channel.shape,"#print2")
    return split_channel
    