# Simple speaker recognizer
This is a simple speaker recognizer.  It allows to save audio recordings of people
voices and then use them to train HMM models.  This program uses Hidden Markov Model
with Gaussian mixture emissions from hmmlearn library.

This application can easily differentiate between the voices based on the same
phrase.  However, it also was quite successful (limited amount of testing)
to differentiate voices with different phrases.

# How to use it
Clone the repo to your local system.  audio.tgz contains sample recordings for
2 females and 3 males.  Start by untaring it:

```
tar -xvzf audio.tgz
```

Next step is to create models for each person's voice:

```
mkdir models
mkdir models\paige
mkdir models\jen
mkdir models\rob
mkdir models\harsha
mkdir models\chris
```

Now you are ready to train models:

```
./speaker_recognizer.py --train-folder audio/paige/ --output-model-folder models/paige/paige.model --n-mix 30
./speaker_recognizer.py --train-folder audio/jen/ --output-model-folder models/jen/jen.model --n-mix 30
./speaker_recognizer.py --train-folder audio/rob/ --output-model-folder models/rob/rob.model --n-mix 30
./speaker_recognizer.py --train-folder audio/harsha/ --output-model-folder models/harsha/harsha.model --n-mix 30
./speaker_recognizer.py --train-folder audio/chris/ --output-model-folder models/chris/chris.model --n-mix 30
```

To test try this:

```
./speaker_recognizer.py --model-folder models/ --test-with-file audio/paige/paige9.wav
./speaker_recognizer.py --model-folder models/ --test-with-file audio/paige/jen9.wav
./speaker_recognizer.py --model-folder models/ --test-with-file audio/paige/rob9.wav
./speaker_recognizer.py --model-folder models/ --test-with-file audio/paige/harsha9.wav
./speaker_recognizer.py --model-folder models/ --test-with-file audio/paige/chris9.wav
```

You can also test directly by speaking your phrase with this command:

```
./speaker_recognizer.py --model-folder models/
```
