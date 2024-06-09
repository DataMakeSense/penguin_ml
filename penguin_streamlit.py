import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title('Penguin Classifier')
st.write('This app uses 6 inputs to predict the specific of penguin')

password_guess = st.text_input('What is your password?')
if password_guess != st.secrets['streamlit_password']:
    st.stop()

penguin_file = st.file_uploader('Upload your own penguin data')
if penguin_file is None:
    rf_pickle = open('random_forest_penguin.pickle', 'rb')
    map_pickle = open('output_penguin.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_penguin_mapping = pickle.load(map_pickle)
    rf_pickle.close()
    map_pickle.close()
else:
    penguin_df = pd.read_csv(penguin_file)
    penguin_df = penguin_df.dropna()
    output = penguin_df['species']
    features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
    features = pd.get_dummies(features)
    output, unique_penguin_mapping = pd.factorize(output)
    x_train, x_test, y_train, y_test = train_test_split(features, output, test_size = 0.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train.values, y_train)
    y_predict = rfc.predict(x_test.values)
    score = accuracy_score(y_predict, y_test)
    st.write(f'Predict Score is {score:.2f}')

with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options = ['Biscoe', 'Dream', 'Torgenson'])
    sex = st.selectbox('Sex', options = ['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value = 0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value = 0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value = 0)
    body_mass = st.number_input('Body Mass (g)', min_value = 0)

    st.form_submit_button()

island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgenson':
    island_torgenson = 1

sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, 
                               body_mass, island_biscoe, island_dream, 
                               island_torgerson, sex_female, sex_male]])

predicted_species = unique_penguin_mapping[new_prediction][0]

st.write(f'We predict your penguin is of the {predicted_species} species')

st.image('feature_importance.png')

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['bill_length_mm'], hue = penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['bill_depth_mm'], hue = penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x = penguin_df['flipper_length_mm'], hue = penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)