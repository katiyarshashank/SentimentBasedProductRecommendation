# Import Required Package
import pickle as pkl

# loading all pickle files
xgb = pkl.load(open('models/Xgboost.pkl', 'rb'))
tfidf = pkl.load(open('models/tfidf.pkl', 'rb'))
transform = pkl.load(open('dataset/transform.pkl', 'rb'))
user_recom = pkl.load(open('models/user_recommendation.pkl', 'rb'))


def recommendation(user_input):
    try:
        flag = True
        data = user_recom.loc[user_input].sort_values(ascending=False)[0:20].index

    except:
        flag = False
        data = 'User not present in dataset'

    return flag, data


def sentiment(prod_list):
    df = transform[transform.name.isin(prod_list)]
    features = tfidf.transform(df['text'])
    pred = xgb.predict(features)
    predictions = [round(value) for value in pred]
    df['predicted'] = predictions

    return df[df['predicted'] == 1][['name', 'brand', 'categories']].drop_duplicates()[:5].reset_index(drop=True)
