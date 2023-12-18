from flask import Flask, render_template, request
from collections import Counter
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,accuracy_score,confusion_matrix

import nshap
from pretty_html_table import build_table

app = Flask(__name__)

import pandas as pd
from ucimlrepo import fetch_ucirepo

hcv_data = fetch_ucirepo(id=571) 
  
# data (as pandas dataframes) 
X = hcv_data.data.features 
y = hcv_data.data.targets 
df = pd.concat([X,y], axis=1)
df= df.rename(columns={"CGT": "GGT"})

df['Category'] = df['Category'].replace({'0=Blood Donor': 'Control', '0s=suspect Blood Donor':'Control' , '1=Hepatitis': "Hepatitis", '2=Fibrosis': "Fibrosis", '3=Cirrhosis': "Cirrhosis"})
df['CategorySimple'] = df['Category'].replace({'Control': 'Control', 'Hepatitis':'Hepatitis' , 'Fibrosis': "Hepatitis", 'Cirrhosis': "Hepatitis"})

y['Category'] = y['Category'].replace({'0=Blood Donor': 'Control', '0s=suspect Blood Donor':'Control' , '1=Hepatitis': "Hepatitis", '2=Fibrosis': "Fibrosis", '3=Cirrhosis': "Cirrhosis"})
y['CategorySimple'] = y['Category'].replace({'Control': 'Control', 'Hepatitis':'Hepatitis' , 'Fibrosis': "Hepatitis", 'Cirrhosis': "Hepatitis"})

cat_columns = X.select_dtypes(["object"]).columns
X[cat_columns] = X[cat_columns].apply(lambda x: x.astype("category").cat.codes)

cat_columns = y.select_dtypes(["object"]).columns
y[cat_columns] = y[cat_columns].apply(lambda x: x.astype("category").cat.codes)

import plotly
import plotly.graph_objs as go
import plotly.express as px

def create_scatter_summary():
    fig = px.scatter_matrix(df.drop(["Category","CHOL","ALP","Sex",
                                     "PROT","CREA","GGT"],axis=1))
    return plotly.io.to_html(fig,full_html=False)#,default_width='40%',
                             #default_height='75%')

def create_jitter_summary(variable):
    fig = px.strip(df, y=variable,color='CategorySimple',
                stripmode='overlay',title="Jitter plot of "+variable)
    fig.update_layout(paper_bgcolor="LightBlue")

    return plotly.io.to_html(fig,full_html=False,default_width='40%',
                             default_height='75%')
    
def create_ASP_missing_pie(values,title=""):
    labels = ['Control','Hepatitis','Fibrosis','Cirrhosis']
    colors = ['blue','red','green','purple']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig.update_traces(marker=dict(colors=colors),textinfo='label+percent',
                      title=title)
    return plotly.io.to_html(fig,full_html=False,default_width='40%',
                             default_height='75%')

def create_imbalanced_pie(values,title=""):
    labels = ['Hepatitis','Control']
    colors = ['light-blue','light-red']
    fig = go.Figure(data=[go.Pie(labels=labels, values=values,
                                 pull=[0, 0.2])])
    fig.update_traces(marker=dict(colors=colors),textinfo='label+percent',
                      title=title)
    return plotly.io.to_html(fig,full_html=False,default_width='40%',
                             default_height='75%')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dataset-information")
def dataset_information():
    return render_template("dataset-information.html")

@app.route("/preprocessing")
def preprocessing():
    import sys
    return render_template("preprocessing.html",table = df.to_html(table_id="initial"))

@app.route("/summary-statistics")
def summary_statistics():
    scat_summ = create_scatter_summary()
    
    
    return render_template("summary-statistics.html",
                           SCAT_SUM = scat_summ)

@app.route("/visualisation-analysis")
def visualisation():
    alt_fig = create_jitter_summary("BIL")
    ast_fig = create_jitter_summary("AST")
    
    imbal_men = create_imbalanced_pie([22,216])
    imbal_f = create_imbalanced_pie([53,324])
    na_pie = create_ASP_missing_pie(values = [0,3,6,9])
    notna_pie = create_ASP_missing_pie(values = [540,21,12,24])
    return render_template("visualisation-analysis.html",
                           ALT = alt_fig,
                           AST = ast_fig,
                           NAPIE = na_pie,
                           NOTNAPIE = notna_pie,
                           IMBAL_MEN=imbal_men,
                           IMBAL_WOMEN=imbal_f)

@app.route("/classification-analysis")
def classification():
    return render_template("classification-analysis.html")

@app.route("/render-analysis", methods=["GET"])
def render_analysis():
    import xgboost
    from sklearn.linear_model import RidgeClassifierCV,LogisticRegressionCV
    from sklearn.linear_model import RidgeClassifier,LogisticRegression
    import sys
    from sklearn.model_selection import StratifiedKFold
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    from sklearn.svm import SVC

    
    simple = request.args.get("simple")
    y2 = y.copy(deep=True)
    if simple:
        y2['Category'] = y["CategorySimple"]
    y_cur=y2['Category']
    
    method = request.args.get("method")
    #print(method, file=sys.stderr)
    
    confusion = request.args.get("confusion")
    
    
    tunebool = request.args.get("tune")
    
    #---- outputs ---
    outputs = {}
    
    if method == "xgboost":
            model = xgboost.XGBClassifier(
                tree_method="hist",enable_categorical=True
            )
            X_train, X_test, y_train, y_test = train_test_split(X,y_cur,test_size=0.333)
            model.fit(X_train,y_train)
            print(y_test,file=sys.stderr)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y_cur,test_size=0.2)
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(X_train)
        IterativeImputer(random_state=0)
        X_train = pd.DataFrame(imp.transform(X_train),columns = X_train.columns)
        
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(X_test)
        IterativeImputer(random_state=0)
        X_test = pd.DataFrame(imp.transform(X_test),columns = X_test.columns)
    
        if tunebool is not None and int(tunebool) ==1:
            # need to do cross validation unless XGBoost
            if method == "ridge":
                model = RidgeClassifierCV(cv=StratifiedKFold(5))
                model.fit(X_train,y_train)
            elif method == "svc":
                model = SVC(probability=True)
                model.fit(X_train,y_train)
            elif method == "logistic":
                #stratified by default
                model = LogisticRegressionCV(cv=5)
                model.fit(X_train,y_train)
            
        else:
            # need to get parameter value
            tuneval = float(request.args.get("hypervalue"))
            if method == "ridge":
                model = RidgeClassifier(alpha=tuneval)
                model.fit(X_train,y_train)
            elif method == "svc":
                model = SVC(C=tuneval)
                model.fit(X_train,y_train)
            elif method == "logistic":
                model = LogisticRegression(penalty='l2',C=tuneval)
                model.fit(X_train,y_train)
        
    # --- accuracy 
    accuracy = balanced_accuracy_score(y_test, model.predict(X_test))*100
    outputs['accuracy'] =  accuracy
    print(accuracy, file=sys.stderr)
    
    # --- confusion matrix
    if confusion:   
        if simple: 
            labels_inner = ["Control","Hepatitis"]
        else:
            labels_inner = ["Control","Hepatitis","Fibrosis","Cirrhosis"]
        labels=dict(y="True Value", x="Predicted Value")
        
        confusion_mat = confusion_matrix(y_test, model.predict(X_test),
                                         labels=y2['Category'].unique())
        print(confusion_mat, file=sys.stderr)
        confusion_mat=100*confusion_mat/confusion_mat.sum(axis=0)
        confusion_mat = np.nan_to_num(confusion_mat,0)
        fig  = px.imshow(confusion_mat,
                  labels=labels,
                  x=labels_inner,
                  y=labels_inner)
        outputs['confusion'] = fig.to_html(full_html=False,default_width='40%',
                             default_height='75%')
   
   
    
    
    
    return render_template("render-analysis.html",**outputs)


# @app.route("/state/<string:name>")
# def state(name):
#     f = df.loc[df.State == name,['State','Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']]
#     rate = df['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']
#     n_rate = (f['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000'] - np.nanmean(rate))/np.nanstd(rate)
#     hl = np.array([])
#     for val in n_rate:
#         if(val<-1):
#             hl = np.append(hl,'low')
#         elif(val>1):
#             hl = np.append(hl,'high')
#         else:
#             hl = np.append(hl,'medium')
    
#     f['Relative Incidence Rate'] = hl
#     return f"""
#     <html>
#     <body>
#     {str(f.to_json(orient='records'))}
#     </body>
#     </html>
#     """

# @app.route("/info", methods=["GET"])
# def info():
#     usertext = request.args.get("state")
#     if not usertext.lower() in df.State.str.lower().unique():
#         return f"""
#         <html>
#         <body>
#         <p><b>You entered an incorrect State name</b></p>
#         <a href="http://127.0.0.1:5000/">Home Page</a>
#         </body>
#         </html>
#         """
#     usertext = usertext.lower().capitalize()
#     f = df.loc[df.State == usertext,['State','Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']]
#     rate = df['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000']
#     n_rate = (f['Age-Adjusted Incidence Rate([rate note]) - cases per 100,000'] - np.nanmean(rate))/np.nanstd(rate)
#     hl = np.array([])
#     for val in n_rate:
#         if(val<-1):
#             hl = np.append(hl,'low')
#         elif(val>1):
#             hl = np.append(hl,'high')
#         else:
#             hl = np.append(hl,'medium')
    
#     f['Relative Incidence Rate'] = hl
#     return f"""
#     <html>
#     <body>
#     <p>{str(f.to_json(orient='records'))}</p>
#     <a href="http://127.0.0.1:5000/">Home Page</a>
#     </body>
#     </html>
#     """


if __name__ == "__main__":
    app.run(debug=True)
