from multiprocessing import freeze_support

from feature.textfeatures.topic_modeling import TopicModeling

def main():
    model = TopicModeling()
    model.initialiseModel()
    model.trainAndSaveModel()

main()