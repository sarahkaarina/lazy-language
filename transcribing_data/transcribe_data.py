"""
Transcribing data function

This function uses Whisper to transcribe your audio data and output it as a .csv file.

You can set whisper to whichever model you would like to use, (i.e., tine, base etc).

Author: Sarah K. Crockford
Date: 11/10/2024

For demo notebooks on how to transcribe data, check out my github page:
https://github.com/sarahkaarina/lazy-language
"""

"""
LIBRARY REQUIREMENTS:
"""

import whisper

# for data wrangling:

import pandas as pd
import numpy as np

def transcribe_data(model2use, audio_data, language):
    
    model = whisper.load_model(model2use)

    result = model.transcribe(audio_data, language=language, verbose=True)

    return result

