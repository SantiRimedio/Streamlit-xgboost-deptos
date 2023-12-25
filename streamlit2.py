import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import streamlit as st
import numpy as np
import pickle
from geopy.geocoders import Nominatim
import time

# Cargo el scaler
scaler = load(r'models/standard_scaler_fit.pkl')

# Cargo el modelo predictivo
with open(r"models/xgboost_best_hedonic_model.pkl", 'rb') as file:
    loaded_model = pickle.load(file)

# Titulo
st.title("Valuación de Departamentos")

# Tomo la ubicación
location_text = st.text_input("Dirección de la propiedad", "Buenos Aires, Argentina")

try:
    # Georeferencio la ubicación
    geolocator = Nominatim(user_agent="property_valuation_app")
    location_info = geolocator.geocode(location_text)
    if location_info:
        Latitud = location_info.latitude
        Longitud = location_info.longitude
    else:
        Latitud, Longitud = 0.0, 0.0
except:
    st.error(f"Error: {location_text}. La ubicación es incorrecta.")


# Map Display Section
st.header("Ubicación de la Propiedad")
data = [{'LATITUDE': Latitud, 'LONGITUDE': Longitud}]
df = pd.DataFrame(data)
st.map(data=df, latitude="LATITUDE", longitude="LONGITUDE", zoom=15, size=10)

# Property Details Section
st.header("Detalles de la Propiedad")

PropiedadSuperficieTotal = st.number_input("Superficie Total (m²)", min_value=0, max_value=500, value=50)
Antiguedad = st.number_input("Antiguedad", min_value=0, max_value=100, value=0)
CantidadDormitorios = st.slider("Cantidad de Ambientes", min_value=1, max_value=10, value=2)

# Prediction Section
st.header("Resultados de la Valuación")

if st.button("Calcular Valuación"):
    with st.spinner("Realizando la predicción..."):
        # Predicción
        features = np.array([[Latitud, Longitud, PropiedadSuperficieTotal, CantidadDormitorios, Antiguedad, 2023]])

        try:
            features = scaler.transform(features)
            # Simulating a longer loading time (5 seconds delay)
            time.sleep(5)
            prediction = loaded_model.predict(features)[0]

            # Display prediction with margin of error (10%)
            margin_of_error = 0.1 * prediction
            low_limit = round(max(0, prediction - margin_of_error))
            high_limit = round(prediction + margin_of_error)

            low_limit_str = "{:,.2f}".format(low_limit)
            high_limit_str = "{:,.2f}".format(high_limit)

            success_message = f"""
                <div style="background-color: #baffc0; color: #000000; border: 1px solid #D6E9C6; padding: 15px; border-radius: 4px;">
                    <strong>Valor Estimado:</strong> ${low_limit_str} - ${high_limit_str}
                </div>
            """
            st.markdown(success_message, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error durante la predicción: {e}")
            prediction = 0.0  # or any default value
