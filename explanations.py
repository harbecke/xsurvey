from keras import backend
from keras.models import Model
import numpy as np

from deepexplain.tensorflow import DeepExplain


def create_explanations(model, x_data, y_data):

    with DeepExplain(session=backend.get_session()) as de:
        input_tensor = model.layers[0].input
        fModel = Model(inputs=input_tensor, outputs = model.layers[-2].output)
        target_tensor = fModel(input_tensor)
        
        attributions = dict()
        explanation_keys = [('Sensitivity', 'saliency'), ('Gradient*Input', 
            'grad*input'), ('epsilon-LRP', 'elrp'), ('Occlusion', 'occlusion'),
            ('IntegratedGradients', 'intgrad')]
        for key1, key2 in explanation_keys:
            if key1 == 'Occlusion': window_shape=(3,300)
            else: window_shape=None

            attributions[key1] = de.explain(key2, target_tensor,
                input_tensor, xs=x_data, ys=y_data, window_shape=window_shape)
        
        attributions_summed = dict()
        for key in attributions.keys():
            attributions_summed[key] = np.sum(attributions[key], axis=2)

    return attributions, attributions_summed
