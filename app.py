from flask import Flask, render_template, request, send_file
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import csv



app = Flask(__name__, template_folder = 'template')
model = pickle.load(open("decision_tree_model.pkl", 'rb'))

@app.route('/',methods=['GET'])
def Home():
    return render_template('Index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        
        Age = int(request.form['Year'])
        income = float(request.form["Income per annum"])
        expenditure = float(request.form["monthly expenditure"])
        cars=int(request.form["car ownership"])
        childrenP=int(request.form['Children Present'])
        childrenT=int(request.form['Total Children'])
        
        country = request.form["Country of residence"]
        if(country=='Canada'):
            CountryRegionName_Canada=1
            CountryRegionName_France=0
            CountryRegionName_Germany=0
            CountryRegionName_United_Kingdom=0
            CountryRegionName_United_States=0
        elif(country=="France"):
            CountryRegionName_Canada=0
            CountryRegionName_France=1
            CountryRegionName_Germany=0
            CountryRegionName_United_Kingdom=0
            CountryRegionName_United_States=0
        elif(country=="Germany"):
            CountryRegionName_Canada=0
            CountryRegionName_France=0
            CountryRegionName_Germany=1
            CountryRegionName_United_Kingdom=0
            CountryRegionName_United_States=0
        elif(country=="United Kingdom"):
            CountryRegionName_Canada=0
            CountryRegionName_France=0
            CountryRegionName_Germany=0
            CountryRegionName_United_Kingdom=1
            CountryRegionName_United_States=0
        elif(country=="United States"):
            CountryRegionName_Canada=0
            CountryRegionName_France=0
            CountryRegionName_Germany=0
            CountryRegionName_United_Kingdom=0
            CountryRegionName_United_States=1
        elif(country=="Australia"):
            CountryRegionName_Canada=0
            CountryRegionName_France=0
            CountryRegionName_Germany=0
            CountryRegionName_United_Kingdom=0
            CountryRegionName_United_States=0
        
        occupation = request.form['Occupation']
        if(occupation=='Managment'):
            Occupation_Management=1
            Occupation_Manual=0
            Occupation_Professional=0
            Occupation_Skilled_Manual=0
        elif(occupation=="Manual"):
            Occupation_Management=0
            Occupation_Manual=1
            Occupation_Professional=0
            Occupation_Skilled_Manual=0
        elif(occupation=="Professional"):
            Occupation_Management=0
            Occupation_Manual=0
            Occupation_Professional=1
            Occupation_Skilled_Manual=0
        elif(occupation=="Skilled Manual"):
            Occupation_Management=0
            Occupation_Manual=0
            Occupation_Professional=0
            Occupation_Skilled_Manual=1
            
        education = request.form['Education Level']
        if(education=='Graduate Degree'):
            Education_Graduate_Degree=1
            Education_High_School=0
            Education_Partial_High_School=0
            Education_Partial_College=0
        elif(education=="High School"):
            Education_Graduate_Degree=0
            Education_High_School=1
            Education_Partial_High_School=0
            Education_Partial_College=0
        elif(education=="Partial High School"):
            Education_Graduate_Degree=0
            Education_High_School=0
            Education_Partial_High_School=1
            Education_Partial_College=0
        elif(education=="Partial College"):
            Education_Graduate_Degree=0
            Education_High_School=0
            Education_Partial_High_School=0
            Education_Partial_College=1
        elif(education=="Bachelors"):
            Education_Graduate_Degree=0
            Education_High_School=0
            Education_Partial_High_School=0
            Education_Partial_College=0
        
        gender=request.form['gender']
        if(gender=='Male'):
            Gender_M=1
        else:
            Gender_M=0
            
        Marital_status=request.form['Marital Status']
        if(Marital_status=='S'):
            MaritalStatus_S=1
        else:
            MaritalStatus_S=0
            
            
        prediction=model.predict([[cars,childrenP,childrenT,income,expenditure,Age,Education_Graduate_Degree,Education_High_School,Education_Partial_College,Education_Partial_High_School,Occupation_Management,Occupation_Manual,Occupation_Professional,Occupation_Skilled_Manual,Gender_M,MaritalStatus_S,CountryRegionName_Canada,
            CountryRegionName_France,CountryRegionName_Germany,CountryRegionName_United_Kingdom,CountryRegionName_United_States]])
        
        if prediction==0:
            return render_template('Index.html',prediction_text="Does not buy bike")
        else:
            return render_template('Index.html',prediction_text="does buy bike")
    else:
        return render_template('Index.html')
        
@app.route("/data", methods=["GET", "POST"])     
def data():
    if request.method == "POST":
        df = pd.read_csv(request.files.get('file'))
        df = pd.get_dummies(df, columns=["Education","Occupation","Gender","MaritalStatus","CountryRegionName"])
        
        if "CountryRegionName_United States" not in df.columns:
            df["CountryRegionName_United States"] = 0
        if "CountryRegionName_United Kingdom" not in df.columns:
            df["CountryRegionName_United Kingdom"] = 0
        if "CountryRegionName_Australia" not in df.columns:
            df["CountryRegionName_Australia"] = 0
        if "CountryRegionName_Canada" not in df.columns:
            df["CountryRegionName_Canada"] = 0
        if "CountryRegionName_Germany" not in df.columns:
            df["CountryRegionName_Germany"] = 0
        if "CountryRegionName_France" not in df.columns:
            df["CountryRegionName_France"] = 0
        if "Education_Bachelors" not in df.columns:
            df["Education_Bachelors"] = 0
        if "Education_Graduate Degree" not in df.columns:
            df["Education_Graduate Degree"] = 0
        if "Education_High School" not in df.columns:
            df["Education_High School"] = 0
        if "Education_Partial College" not in df.columns:
            df["Education_Partial College"] = 0
        if "Education_Partial High School" not in df.columns:
            df["Education_Partial High School"] = 0
        if "Occupation_Management" not in df.columns:
            df["Occupation_Management"] = 0
        if "Occupation_Manual" not in df.columns:
            df["Occupation_Manual"] = 0
        if "Occupation_Professional" not in df.columns:
            df["Occupation_Professional"] = 0
        if "Occupation_Skilled Manual" not in df.columns:
            df["Occupation_Skilled Manual"] = 0
        if "MaritalStatus_S" not in df.columns:
            df["MaritalStatus_S"] = 0
        
        
        
        df = df[['NumberCarsOwned',
       'NumberChildrenAtHome', 'TotalChildren', 'YearlyIncome',
       'AveMonthSpend','Age',
       'Education_Graduate Degree', 'Education_High School',
       'Education_Partial College', 'Education_Partial High School',
       'Occupation_Management', 'Occupation_Manual',
       'Occupation_Professional', 'Occupation_Skilled Manual',
       'Gender_M', 'MaritalStatus_S',
       'CountryRegionName_Canada',
       'CountryRegionName_France', 'CountryRegionName_Germany',
       'CountryRegionName_United Kingdom', 'CountryRegionName_United States']]
        
        df.rename(columns = {'Education_Graduate Degree':'Education_Graduate_Degree', 'Education_High School':'Education_High_School', 'Education_Partial College': 'Education_Partial_College', 'Education_Partial High School':'Education_Partial_High_School', 'Occupation_Skilled Manual':'Occupation_Skilled_Manual', 'CountryRegionName_United Kingdom':'CountryRegionName_United_Kingdom', 'CountryRegionName_United States':'CountryRegionName_United_States'}, inplace=True)
        df = np.asarray(df)
        predict_multiple = model.predict(df)
        lst = []
        for i in range(len(predict_multiple)):
            if predict_multiple[i] == 1:
                lst.append("yes")
            else:
                lst.append("no")
        with open("information.csv", "w", newline = "") as csvfile:
            fieldnames = ["s:no", "decision"]
            
            thewriter = csv.DictWriter(csvfile, fieldnames = fieldnames)
            thewriter.writeheader()
            dec_count = 0
            for dec in lst:
                dec_count += 1
                thewriter.writerow({"s:no" : dec_count, "decision" : dec})
        return render_template("Index.html", predict_multiple= lst)

@app.route("/download")      
def download_file():
    p = "information.csv"
    return send_file(p, as_attachment=True)
   

if __name__== "__main__":
    app.run(debug=True)