from flask import Flask,request,jsonify
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('random_forest_model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def index():
    return "Hello world"
@app.route('/predict',methods=['POST'])
def predict():
    counties_and_subcounties=pd.read_csv('C:/Users/nyaga/Desktop/project model/training_data.csv')
    df = pd.read_csv('C:/Users/nyaga/Desktop/project model/training_data.csv')
    df['Quantity Ton LR 2018'] = df['Quantity Ton LR 2018'].str.replace(',', '').astype(float)
    df['Area Ha LR 2018'] = df['Area Ha LR 2018'].str.replace(',', '').astype(float)
    df['yield_per_ha'] = df['Quantity Ton LR 2018'] / df['Area Ha LR 2018']
    df=df.drop('Quantity Ton LR 2018', axis=1)
    all_county_names = counties_and_subcounties['COUNTY'].unique()
    all_subcounty_names = counties_and_subcounties['SUBCOUNTY'].unique()
    df = pd.get_dummies(df, columns=['COUNTY', 'SUBCOUNTY'],)
    column_order = df.columns.drop('yield_per_ha')
    county = request.form.get('county')
    subcounty = request.form.get('subcounty')
    area = request.form.get('area')
    temperature = request.form.get('temperature')
    humidity = request.form.get('humidity')
    precipitation = request.form.get('precipitation')
    

    input_data = pd.DataFrame({
    'COUNTY':['BARINGO'],
    'SUBCOUNTY':['Eldama Ravine'],
    'Area Ha LR 2018':[200],
    'Temperature': [30],
    'Humidity': [65.0],
    'precipitation': [60.0],
     # One-hot encoded subcounty variable
    # Include all relevant features
})
# Remove the available 'county' and 'subcounty' names from the respective arrays
    if 'COUNTY' in input_data:
        available_county_name = input_data['COUNTY'].iloc[0]
        all_county_names = all_county_names[all_county_names != available_county_name]

    if 'SUBCOUNTY' in input_data:
        available_subcounty_name = input_data['SUBCOUNTY'].iloc[0]
        all_subcounty_names = all_subcounty_names[all_subcounty_names != available_subcounty_name]
        
        
    county = pd.DataFrame({'COUNTY_' + available_county_name: [1]}, index=input_data.index)
    subcounty = pd.DataFrame({'SUBCOUNTY_' + available_subcounty_name: [1]}, index=input_data.index)
    # Create DataFrames for missing columns
    missing_county_columns = pd.DataFrame(0, columns=['COUNTY_' + county_name for county_name in all_county_names], index=input_data.index)
    missing_subcounty_columns = pd.DataFrame(0, columns=['SUBCOUNTY_' + subcounty_name for subcounty_name in all_subcounty_names], index=input_data.index)

    # Concatenate the missing columns with the original DataFrame
    input_data = pd.concat([input_data,county, subcounty, missing_county_columns, missing_subcounty_columns], axis=1)

    # Now, the input_data DataFrame includes missing columns for both 'county' and 'subcounty' after removing available names
    input_data.drop(columns=['COUNTY', 'SUBCOUNTY'], inplace=True)

    # Reorder the columns in the input data to match the training data
    input_data = input_data[column_order]
    result = model.predict(input_data)[0]
    
    return jsonify({'yield in kg per ha':str(result)})
if __name__ == '__main__':
    app.run(debug=True)