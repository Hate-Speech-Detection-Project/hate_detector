# Hate-Speech Detection Framework

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
