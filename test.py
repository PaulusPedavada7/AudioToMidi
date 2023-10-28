import tensorflow as tf

from basic_pitch.inference import predict_and_save
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

basic_pitch_model = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))




input = 'test1.wav'

output = 'test_output.MIDI'

predict_and_save(
    input,
    output,
    True,
    False,
    False,
    False,
    )






