FROM python

COPY . /code
ENV NLTK_DATA ./nltk
WORKDIR /code
RUN pip install -r requirements.txt
RUN mkdir -p nltk
RUN python -m nltk.downloader stopwords

CMD python .
