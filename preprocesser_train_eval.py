import os
import numpy as np
import librosa
from sklearn.model_selection import StratifiedKFold
from hmmlearn import hmm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import itertools

'''
The following statement defines a scaler object for transforming data
to standardized form by removing the mean and scaling unit variance.
'''
scaler = StandardScaler()

'''
We define the relative data path to the Hugging Face data set located
in the project repository.
'''
path = 'digit_dataset'

'''
We define a digit map that maps each digit word to a digit numeric value
such that the model can use the numeric values as training and testing labels.
'''
digit_map = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9
}

'''
We define a routine that utilizes the librosa module
to load the .wav audio files for each digit label and
to also extract the Mel-frequency cepstral coefficients.
The y contains the waveform data and the sampling rate is set
to None to preserve the original sampling rate. The number of
Mel-frequency cepstral coefficients in the extraction function
is set to 13 which is common for speech analysis applications.
We return the mean of the first axis of the transposed
Mel-frequency cepstral coefficients matrix representing the
mean of an individual frame. This function essentially takes
the audio file input and returns the mean values of the
13 Mel-frequency cepstral coefficients over the entire
audio file.
'''
def feature_extraction_engine(file_path):
    y_form, sampling_rate = librosa.load(file_path, sr=None)
    coefficients = librosa.feature.mfcc(y=y_form, sr=sampling_rate, n_mfcc=13)
    return np.mean(coefficients.T, axis=0)


'''
We define the data loader routine which extracts the 
Mel-frequency cepstral coefficients from the path .wav file and also extracts
the digit label and return the np array for the file features and a corresponding
label for the digit labels.
'''
def dataset_loader():
    X = []
    y = []
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            digit_var = filename.split('_')[0]
            if digit_var in digit_map:
                X.append(feature_extraction_engine(os.path.join(path, filename)))
                y.append(digit_map[digit_var])
    return np.array(X), np.array(y)

'''
The following routine returns the prediction matrix using the trained
Hidden Markov Model on the feature array of Mel-frequency cepstral coefficients
for the entire testing dataset. The routine iterates over each feature in the 
feature set and initializes the log_likelihood array to store the log_likelihood scores
. The model then calculates the log-likelihood score of the feature using the score
routine and return -inf for valueError occurences. The highest log-likelihood score
is then appended with the label in its place and converts the result to numpy arrays.
'''
def hmm_prediction_engine(models, X):
    prediction_result = []
    for x in X:
        log_likelihood_score_array = []
        for model in models.values():
            try:
                log_likelihood_score_array.append(model.score([x]))
            except ValueError:
                log_likelihood_score_array.append(float('-inf'))
        prediction_result.append(max(models.keys(),
                                     key=lambda k: log_likelihood_score_array[list(models.keys()).index(k)]))
    return np.array(prediction_result)


'''
This block initializes the transition matrix and initial state probability
vector for a hidden markov model using 5 hidden state components. The initial
transition matrix is initializes as a square 5 x 5 matrix with each entry
initialized to 1.0 / component_count. This gives each node equal probability of
transitioning to any other component. The initial_statprob creates the initial
state probability vector with length 5 with nodes initialized to 1.0 / component
count for equal probability of being the initial state.
'''
component_count = 5
transmat_initial = np.full((component_count, component_count), 1.0 / component_count)
startprob_initial = np.full(component_count, 1.0 / component_count)

'''
We load the dataset with the feature set and the label set and apply
scalar standardization to the feature set. We then initialize the
Stratified K-Folds cross validation object to shuffle the data before
splitting on a random_state seed with initialized empty lists for the
y_test and y_pred.
'''
Feature_Set, y = dataset_loader()
Feature_Set = scaler.fit_transform(Feature_Set)

stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_test_arr = []
y_pred_arr = []

'''
We iterate over the stratified_k_fold train and test indices for the feature
set and initiize for each fold the training, testing feature set as well as the
label training and testing set. We then create the hmm_model set and iterate
over each digit label pair and initialize the gaussian hidden markov model
with the specific param values for the components, covariance and iteration count.
Then we initialize the transition probabilities and starting probabilities. We
finally filder the training data to include samples for the appropriate current
digit label. When training data exists for the current digit label we fit the
Gaussian HMM model to the data and store it in the dictionary. If there is no
training data for the current digit label we output a debug statement.
Then we use the hmm_prediction_engine to use the model to predict over the
test feature set and extend the two y_test_arr and y_pred_arr arrays for each
fold.
'''
for train_idx, test_idx in stratified_k_fold.split(Feature_Set, y):
    X_train, X_test = Feature_Set[train_idx], Feature_Set[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    hmm_model_arr = {}

    for digit_var, label in digit_map.items():
        model = hmm.GaussianHMM(n_components=component_count, covariance_type='diag', n_iter=1000)
        model.transmat_ = transmat_initial
        model.startprob_ = startprob_initial
        if len(X_train[y_train == label]) > 0:
            model.fit(X_train[y_train == label])
            hmm_model_arr[label] = model
        else:
            print(f"No data found for digit {digit_var}")

    y_test_arr.extend(y_test)
    y_pred_arr.extend(hmm_prediction_engine(hmm_model_arr, X_test))

print("Classification Report:")
print(classification_report(y_test_arr, y_pred_arr, target_names=digit_map.keys()))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_arr, y_pred_arr))

'''
We define a helper function to generate the confusion matrix generator function
that takes the counts or proportions of true positive false positive etc. predictions
. In addition we pass in the class labels, the boolean parameter for normalization
and the tile, cmap properties. The function uses the normalize_bool parameter to
normalize matrix. The confusion matrix is displayed as an image plot using
plt.imshow() with interpolation to the nearest. Class labels are set on the x and y
values. The plot is labelled withing the routine. After the confusion matrix is
displayed as an image plot with fmt formatting.
'''
def confusion_matrix_generator(confusion_mat, classes, normalize_bool=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize_bool:
        confusion_mat = confusion_mat.astype('float') / confusion_mat.sum(axis=1)[:, np.newaxis]
    plt.imshow(confusion_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    for i, j in itertools.product(range(confusion_mat.shape[0]), range(confusion_mat.shape[1])):
        plt.text(j, i, format(confusion_mat[i, j], '.2f' if normalize_bool else 'd'),
                 horizontalalignment="center",
                 color="white" if confusion_mat[i, j] > confusion_mat.max() / 2. else "black")
    plt.ylabel('Ground True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

'''
We plot the figure and the confusion matrix.
'''
conf_matrix = confusion_matrix(y_test_arr, y_pred_arr)
plt.figure(figsize=(10, 7))
confusion_matrix_generator(conf_matrix, classes=list(digit_map.keys()), title='Confusion Matrix')
plt.show()
