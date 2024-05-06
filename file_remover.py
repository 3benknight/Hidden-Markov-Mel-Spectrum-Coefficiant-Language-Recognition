import os
'''
This script refactors the Hugging Face digit dataset to only include .wav
files for digit audio files in English. The dataset contains 10 languages
with 50 audio files for each digit in each language. We trim the dataset to
only include the English language .wav audio files for digits 0-9.
'''
path = 'digit_dataset'

'''
This is to only raise an exception if the relative path to digit_dataset does
not exit.
'''
if not os.path.exists(path):
    raise FileNotFoundError(f"The directory {path} is non existant.")

'''
This is to ensure that the language retained directory only contains english
but is expandable for future iterations.
'''
retained_language_set = {'en'}

'''
We define the language parser to split the .wav file names to extract
the language code from the file name.
'''
def language_parser(filename):
    return filename.split('_')[1]

'''
We iterate over all the files in the os listdir iterable and check to
see if the language in the filename is not in retained_language_set. If
it is not in the set then it is removed from the directory.
'''
for filename in os.listdir(path):
    if filename.endswith('.wav'):
        language_var = language_parser(filename)
        if language_var not in retained_language_set:
            os.remove(os.path.join(dataset_path, filename))
            print(f"Removing: {file_path}")

print("Repository refactored to include only english language data.")
