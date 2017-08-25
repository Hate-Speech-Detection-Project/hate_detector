# Hate-Speech Detection Framework

### Run the framework using Docker
~~~
docker build -t hate_speech_framework .
docker run -v .:/code -t hate_speech_framework #  "-v .:/code" is optinal but lets you change code / add data and run the latest code
~~~


### Run the framework directly
~~~
# Setup virtualenv with python 3
python3 -m venv venv
source venv/bin/activate # assumes using bash

# Install requirements
pip install -r requirements.txt

# Run framework
python .
~~~
