import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
from sklearn.model_selection import train_test_split
import streamlit as st
import numpy as np
import pickle
from geopy.geocoders import Nominatim

# Cargo el scaler
scaler = load(r'models/standard_scaler_fit.pkl')

# Cargo el modelo predictivo
with open(r"models/xgboost_best_hedonic_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

# Titulo
st.title("Valuación de Departamentos")

# Tomo la ubicación
location_text = st.text_input("Dirección de la propiedad", "Buenos Aires, Argentina")


# Georeferencio la ubicación
geolocator = Nominatim(user_agent="property_valuation_app")
location_info = geolocator.geocode(location_text)
if location_info:
    Latitud = location_info.latitude
    Longitud = location_info.longitude
else:
    Latitud, Longitud = 0.0, 0.0

# Ploteo la ubicación ingresada
data = [{'LATITUDE': Latitud, 'LONGITUDE': Longitud}]
df = pd.DataFrame(data)
st.map(data=df,latitude="LATITUDE", longitude="LONGITUDE", zoom=15)

# Ingreso los otros datos
PropiedadSuperficieTotal = st.number_input("Metros cuadrados", min_value=0, max_value=500, value=50)

Antiguedad = st.number_input("Antiguedad", min_value=0, max_value=100, value=0)

CantidadDormitorios = st.slider("Cantidad de Ambientes", min_value=1, max_value=10, value=2) + 1

AÑO = 2023

# Predicción
features = np.array([[Latitud, Longitud, PropiedadSuperficieTotal, CantidadDormitorios, Antiguedad, AÑO]])
features = scaler.transform(features)
prediction = loaded_model.predict(features)[0]

# Display prediction
st.subheader(f"Valuación ${prediction:,.2f}")

#%%
