import Pandas as pandas;
import Chefboost as Chefboost;



def Build_DT(dataframe):
    dataframe = pd.read_csv("./data/question.txt");
    print(dataframe.head());
    config ={'algorithm':'ID3'};
    model = Chefboost.fit(dataframe,config);
    ClassifyInstance(dataframe,model);


def ClassifyInstance(dataframe):
    for index,instance in dataframe.iterrows():
            prediction = Chefboost.predict(model,instance)
            return(prediction);



dataframe = pd.read_csv("./data/question.txt");
Build_DT(dataframe);





