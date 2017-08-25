# Hate-Speech Detection Framework

![](https://i.imgur.com/GzyofH0.gif)

### Run the framework using Docker
~~~bash
docker build -t hate_speech_framework .
docker run -t hate_speech_framework
~~~


### Run the framework directly
~~~bash
# Setup virtualenv with python 3
python3 -m venv venv
source venv/bin/activate # assumes using bash

# Install requirements
pip install -r requirements.txt

# Install nltk data
mkdir -p nltk
NLTK_DATA=./nltk
python -m nltk.downloader stopwords

# Run framework
python .
~~~

### Add datasets
- add a folder to /data containing a `test.csv` and a `train.csv` (look at the test folder for samples)
- add the name of the dataset folder to the __main__.py (line 49)
