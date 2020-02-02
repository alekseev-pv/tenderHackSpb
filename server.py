import os
import numpy as np
import requests

from sklearn.externals import joblib

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask, request, jsonify

from flask_cors import CORS

import re 

def has_cyr(text):
    return bool(re.search('[а-яА-Я]', text))

def has_lat(text):
    return bool(re.search('[a-zA-Z]', text))



# loading models

filename = 'cat.pkl'
lg = joblib.load(filename)

filename = 'vectorizer.pkl'
vec = joblib.load(filename)

# models by 1-2-3 cats

filename = 'vse11.joblib.pkl'
v1 = joblib.load(filename)

filename = 'vse12.joblib.pkl'
v2 = joblib.load(filename)

filename = 'vse13.joblib.pkl'
v3 = joblib.load(filename)


# custom vec for model 2
filename = 'vec2.pkl'
vec2 = joblib.load(filename)


# loading dicts

exact = np.load("exact.npy", allow_pickle=True).item()

subcat = np.load("subcat.npy", allow_pickle=True).item()




d1 = np.load("ddd1.npy", allow_pickle=True).item()

dict1 = {}
for k, v in d1.items():
    dict1[v] = k


d2 = np.load("ddd2.npy", allow_pickle=True).item()

dict2 = {}
for k, v in d2.items():
    dict2[v] = k
    
d3 = np.load("ddd3.npy", allow_pickle=True).item()

dict3 = {}
for k, v in d3.items():
    dict3[v] = k


# cat dict
cat_dict = ['Работы','Товары','Услуги']


# make some tests 
N = ['Портативная tftft','Замена жесткого диска','Котята в хорошие руки', \
     'Rjejen', 'Котята', "ылоав", "телефон", "ловля рыбы на мормышку", \
     'Абсурдное предложение',"Наушники","Удаление волос","Удаление вирусов","Пианино","Ключ","Кран", "настройка и подключение стиральной машины", "озеленение двора", "наушнки", "сумка для ноутбука"]


def info(s, count=3):
    P = vec.transform([s])
    p = lg.predict(P)[0]
    
    print(s,' - ', cat_dict[p-1])
        
    if (p == 1):  
        PP1 = v1.predict_proba(P)
        #print(PP1)
        #print("predict : ", v1.predict(P), ' - ' ,dict1[v1.predict(P)[0]])
        #print(pd.DataFrame(PP1,columns=dict1.values()).T[0].sort_values(ascending=False).head(3))
        df = pd.DataFrame(PP1,columns=dict1.values()).T[0].sort_values(ascending=False).head(count)
        
    elif(p == 3):
        PP3 = v3.predict_proba(P)
        #print(PP3)
        #print("predict : ", v3.predict(P), ' - ' ,dict3[v3.predict(P)[0]])
        #print(pd.DataFrame(PP3,columns=dict3.values()).T[0].sort_values(ascending=False).head(3))
        df = pd.DataFrame(PP3,columns=dict3.values()).T[0].sort_values(ascending=False).head(count)
        
    else:
        P = vec2.transform([s])
        PP2 = v2.predict(P)
        PP2__ = v2.predict_proba(P)
        #print(PP2__)
        #print("predict : ", v2.predict(P), ' - ' ,dict2[v2.predict(P)[0]])
        #print(pd.DataFrame(PP2__,columns=dict2.values()).T[0].sort_values(ascending=False).head(3))
        df = pd.DataFrame(PP2__,columns=dict2.values()).T[0].sort_values(ascending=False).head(count)
    
    df = df.to_frame()
    df[1] = df.index
    
    print(df)
    
    return df



#for q in N:
#    print("__________")
#    info(q)
    


app = Flask(__name__, static_url_path='/static')

app.config['JSON_SORT_KEYS'] = False

CORS(app)

@app.route("/", methods=['GET','POST'])
def hello():
    if request.method == 'GET':
        start = request.args.get('start', default=0, type=int)
        limit_url = request.args.get('limit', default=20, type=int)
        print("WE HAVE GET!")
        print(request.args)
        print("")
        
# check spell

        if (request.args.get('check')):
            print('check!')
            print(request.args.get('check'))

            
            response = requests.get('https://speller.yandex.net/services/spellservice.json/checkText?text='+request.args.get('check'))
            t = response.json()
            print(t)

            return jsonify(t)
        
# wcat
        if (request.args.get('cat')):
            print('cat!')
            print(request.args.get('cat'))

            P = vec.transform([request.args.get('cat')])
            PP = lg.predict(P)
            
  
            print(cat_dict[PP[0]-1])
    
    
            print(info(request.args.get('cat')))
        
        
            print("!!! ", info(request.args.get('cat')).to_json(force_ascii=False))
    
            dff = info(request.args.get('cat'))
        
            dff_plus = info(request.args.get('cat')+" "+dff.iloc[0,1])
        
            response = requests.get('https://speller.yandex.net/services/spellservice.json/checkText?text='+request.args.get('cat'))
            t = response.json()
            

            
            return jsonify(req = request.args.get('cat'), cat = cat_dict[PP[0]-1], gap1="", a1 = dff.iloc[0,0], b1 = dff.iloc[0,1], \
            a2 = dff.iloc[1,0], b2 = dff.iloc[1,1], a3 = dff.iloc[2,0], b3 = dff.iloc[2,1], gap2="",\
            jobs = lg.predict_proba(P)[0][0], goods = lg.predict_proba(P)[0][1], services = lg.predict_proba(P)[0][2],\
            gap3="",\
            
            c1 = dff_plus.iloc[0,0], d1 = dff_plus.iloc[0,1], \
            c2 = dff_plus.iloc[1,0], d2 = dff_plus.iloc[1,1], c3 = dff_plus.iloc[2,0], d3 = dff_plus.iloc[2,1], gap4="",\
            
                           
             exact = exact[dff.iloc[0,1]], subcategory = subcat[dff.iloc[0,1]],\
                           
                           
            spell = t, mixed_chars = (has_cyr(request.args.get('cat')) and has_lat(request.args.get('cat'))) )
     
    
    

        return jsonify(isError= False,
                        message= "Success",
                        statusCode= 200, r = request.args), 200

    
    # request.form to get form parameter
    
    if request.method == 'POST':
        print("WE HAVE POST!")
        return "POST", 200


if __name__ == "__main__":
        port = int(os.environ.get("PORT", 80))
        app.run(host='0.0.0.0', port=80)
