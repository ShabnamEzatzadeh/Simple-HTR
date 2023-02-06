import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score


# A utility to decode the output of the network
def decode_batch_predictions(pred, characters, labels_to_char):
    pred = pred[:, :-2]
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
    # Iterate over the results and get back the text
    output_text = []
    for res in results.numpy():
        outstr = ''
        for c in res:
            if c < len(characters) and c >= 0:
                outstr += labels_to_char[c]
        output_text.append(outstr)

    # return final text results
    return output_text


def prediction(model, test_data_generator, test_labels, labels_to_char, characters):
    print('Prediction NN model:')
    prediction_model = keras.models.Model(model.get_layer(name='input_data').input,
                                          model.get_layer(name='dense2').output)
    prediction_model.summary()
    prediction_model.save("../models/prediction-model-keras-ours-new.h5", overwrite=True)

    #  Let's check results on some test samples
    pred_tests = []
    gt_tests = []
    for p, (inp_value, _) in enumerate(test_data_generator):
        bs = inp_value['input_data'].shape[0]
        X_data = inp_value['input_data']
        labels = inp_value['input_label']

        preds = prediction_model.predict(X_data)
        pred_texts = decode_batch_predictions(preds, characters, labels_to_char)
        pred_tests = pred_tests + pred_texts

        orig_texts = []
        for label in labels:
            text = ''.join([labels_to_char[int(x)] for x in label])
            orig_texts.append(text)
            gt_tests.append(orig_texts)

        for i in range(bs):
            print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')

        print(len(pred_tests))
        print(len(gt_tests))
        print("******* Total accuracy score is:")
        print(accuracy_score(test_labels, pred_tests))


def prediction_saved_model(test_data_generator, test_labels, labels_to_char, characters):
    print('Prediction NN using pretrained model:')
    prediction_model = load_model("../models/prediction-model-keras-ours-new.h5")
    prediction_model.summary()
    #  Let's check results on some test samples
    pred_tests = []
    gt_tests = []
    for p, (inp_value, _) in enumerate(test_data_generator):
        bs = inp_value['input_data'].shape[0]
        X_data = inp_value['input_data']
        labels = inp_value['input_label']

        preds = prediction_model.predict(X_data)
        pred_texts = decode_batch_predictions(preds, characters, labels_to_char)
        pred_tests = pred_tests + pred_texts

        orig_texts = []
        for label in labels:
            text = ''.join([labels_to_char[int(x)] for x in label])
            orig_texts.append(text)
            gt_tests.append(orig_texts)

        for i in range(bs):
            print(f'Ground truth: {orig_texts[i]} \t Predicted: {pred_texts[i]}')

        print(len(pred_tests))
        print(len(gt_tests))
        print("******* Total accuracy score is:")
        print(accuracy_score(test_labels, pred_tests))
