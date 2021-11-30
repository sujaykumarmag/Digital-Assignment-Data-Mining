#Importing Dependencies
import pandas as pandas;
import chefboost as chef;


def ClassifyInstance(dataframe,config):
    for index,instance in dataframe.iterrows():
            model = chef.fit(dataframe,config);
            prediction = chef.predict(model,instance)
            return(prediction);


def Build_DT(dataframe):
    print(dataframe.head());
    config ={'algorithm':'ID3'};
    ClassifyInstance(dataframe,config);

dataframe = pandas.read_csv("question.txt");
Build_DT(dataframe);





