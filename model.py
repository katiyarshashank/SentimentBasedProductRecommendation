# Import Required Package
import pickle as pkl
import numpy as np

# loading all pickle files
user = pkl.load(open('models/user_recommendation.pkl', 'rb'))
finalmodel = pkl.load(open('models/finalmodel.pkl', 'rb'))
tfidf = pkl.load(open('models/tfidf.pkl', 'rb'))
transform = pkl.load(open('dataset/transform.pkl', 'rb'))


def recommendation(user_input):
    try:
        flag = True
        data = user.loc[user_input].sort_values(ascending=False)[0:20].index
    except:
        flag = False
        data = 'User not present in dataset'

    return flag, data


def sentiment(prod_list):
    df = transform[transform.name.isin(prod_list)]
    features = tfidf.transform(df['text'])
    pred = finalmodel.predict(features)
    predictions = [round(value) for value in pred]
    df['predicted'] = predictions

    groupedDf = df.groupby(['name'])
    product_class = groupedDf['predicted'].agg(mean_class=np.mean)
    df=product_class.sort_values(by=['mean_class'], ascending=False)[:5]
    df['name'] = df.index
    data = df[['name']][:5].reset_index(drop=True)
    data1= data[["name"]].to_numpy()
    np.concatenate(data1, axis=0)
    datalist=[]
    for dataa in data1:
        datalist.append(dataa[0])

    return datalist
