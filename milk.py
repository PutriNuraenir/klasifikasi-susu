import pickle
import streamlit as st
import numpy as np

# membaca model
model = pickle.load(open('milk.sav', 'rb'))
scaler = pickle.load(open('scaler_milk.sav','rb'))

#judul web
st.title('Prediksi Kualitas Susu')

#membagi kolom
col1, col2 = st.columns(2)

with col1 :
    pH = st.number_input('pH Susu')
    Temprature = st.number_input('Suhu Susu')
    Taste = st.number_input('Rasa susu, 0 (Buruk) atau 1 (Baik)')
    Odor = st.number_input('Bau susu , 0 (Buruk) atau 1 (Baik)')

with col2 :
    Fat = st.number_input('Lemak susu, 0 (Rendah) atau 1 (Tinggi)')
    Turbidity = st.number_input('Kekeruhan susu, 0 (Rendah) atau 1 (Tinggi)')
    Colour = st.number_input('Nilai Warna susu yang berkisar antara 240 hingga 255')

# code untuk prediksi
prediction = ''
input_data = (pH, Temprature, Taste, Odor, Fat, Turbidity, Colour)

input_data_as_numpy_array = np.array(input_data)

input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

std_data = scaler.transform(input_data_reshape)

# membuat tombol untuk prediksi
if st.button('Test'):
    milk_prediction = model.predict(std_data)
    if(milk_prediction[0] == 0):
        prediction = 'Kualitas Susu Rendah'
    elif (milk_prediction[0] == 1):
        prediction = 'Kualitas Susu Sedang'
    else:
        prediction = 'Kualitas Susu Tinggi'
    st.success(prediction)
