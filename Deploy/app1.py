import streamlit as st;import seaborn as sns;import matplotlib.pyplot as plt;import pandas as pd
import joblib;from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor;from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,PowerTransformer
from sklearn.pipeline import make_pipeline; from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

st.title('Bioactivity')
st.write("This app uses inputs to predict pIC50 value")
file = st.file_uploader('Upload your own bioactivity data')
if file is None:
  model = joblib.load('model')
else:
  df = pd.read_csv(file).iloc[:,2:]
  df = df.dropna()
  y = df['pIC50']
  x = df.drop("pIC50",axis=1)
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=1)
  numcol = x_train.select_dtypes(include=['int64', 'float64']).columns
  catcol = x_train.select_dtypes(include=['object']).columns
  power = PowerTransformer(method='yeo-johnson')
  rob = RobustScaler(quantile_range=(25.0, 75.0))
  on = OneHotEncoder(sparse=False,drop='first')	
  ct = make_column_transformer((make_pipeline(on), ['bioactivity_class']),
                             (make_pipeline(power,rob),numcol),remainder='passthrough')
  rf = RandomForestRegressor(random_state=15)
  pipeline = make_pipeline(ct,rf,verbose=False)
  pipeline.fit(x_train, y_train)
  y_pred = pipeline.predict(x_test)
  score = round(r2_score(y_pred, y_test), 4)
  st.write('We trained a Random Forest model on these data,'
  ' it has a score of {}. Use the '
  'inputs below to try out the model.'.format(score))
  joblib.dump(pipeline,'model')
  model = joblib.load('model')
  fig, ax = plt.subplots()
  ax = sns.barplot(rf.feature_importances_, x_train.columns)
  plt.title('Which features are the most important for speciesprediction?')
  plt.xlabel('Importance')
  plt.ylabel('Feature')
  plt.tight_layout()
  fig.savefig('feature_importance.png')
  st.write('We used a machine learning (Random Forest) model to '
  'predict the species, the features used in this prediction '
  ' are ranked by relative importance below.')
  st.image('feature_importance.png')

with st.form('user_inputs'): 
  bclass = st.selectbox('bioactivity_class', options=['active', 'inactive']) 
  molwt = st.number_input('MolWt', min_value=0, value=500)
  logp = st.number_input('LogP', min_value=0,value=5)
  numhdonors = st.number_input('NumHDonors', min_value=0,value=3)
  numhacceptors = st.number_input('NumHAcceptors', min_value=0,value=5)
  st.form_submit_button() 
st.write('the user inputs are {}'.format([bclass, molwt, logp,numhdonors, numhacceptors]))
data = [[bclass, molwt, logp,numhdonors, numhacceptors]]
a = pd.DataFrame(data, columns = ['bioactivity_class', 'MolWt','LogP','NumHDonors','NumHAcceptors'])
prediction = model.predict(a)
st.write('We predict your pIC50 is of the {} '.format(prediction))