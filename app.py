# import streamlit as st
# import pickle
# import pandas as pd

# # Load the saved model
# with open('co2_knn_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # App Title
# st.title("🚗 CO2 Emission Predictor")
# st.markdown("Enter car details below to predict **CO2 emissions (g/km)**")

# # Input Fields (exact columns from x_train)
# vehicle_class = st.selectbox("Vehicle Class", [
#     'COMPACT', 'SUV - SMALL', 'MID-SIZE', 'TWO-SEATER', 'MINICOMPACT',
#     'SUBCOMPACT', 'FULL-SIZE', 'STATION WAGON - SMALL', 'SUV - STANDARD',
#     'VAN - CARGO', 'MINIVAN', 'PICKUP TRUCK - SMALL',
#     'STATION WAGON - MID-SIZE', 'PICKUP TRUCK - STANDARD',
#     'VAN - PASSENGER', 'SPECIAL PURPOSE VEHICLE'
# ])

# fuel_type = st.selectbox("Fuel Type", ['Z', 'D', 'X', 'E', 'N'])

# transmission = st.selectbox("Transmission", [
#     'AS6', 'M6', 'AV7', 'AS8', 'A6', 'AM6', 'A8', 'AS7', 'M5',
#     'A4', 'AM7', 'AV', 'A5', 'AV6', 'AS5', 'AM8', 'AS9', 'A7', 'AM9'
# ])

# cylinders = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12, 16])

# engine_size = st.slider("Engine Size (L)", 1.0, 8.0, 2.0, step=0.1)

# fuel_comb = st.slider("Fuel Consumption Comb (L/100 km)", 4.0, 25.0, 9.5, step=0.1)

# # Predict Button
# if st.button("Predict CO2 Emission"):
#     sample = pd.DataFrame([[vehicle_class, fuel_type, transmission,
#                             cylinders, engine_size, fuel_comb]],
#                           columns=['Vehicle Class', 'Fuel Type', 'Transmission',
#                                    'Cylinders', 'Engine Size(L)',
#                                    'Fuel Consumption Comb (L/100 km)'])
#     prediction = model.predict(sample)
#     st.success(f"🌿 Predicted CO2 Emission: **{round(prediction[0], 2)} g/km**")


import streamlit as st
import pickle
import pandas as pd

# Load model, scaler and encoders
with open('co2_knn_model.pkl', 'rb') as f:
    saved = pickle.load(f)

model      = saved['model']
scaler     = saved['scaler']
le_vehicle = saved['le_vehicle']
le_fuel    = saved['le_fuel']
le_trans   = saved['le_trans']

# App Title
st.title("🚗 CO2 Emission Predictor")
st.markdown("Enter car details below to predict **CO2 emissions (g/km)**")

# Input Fields
vehicle_class = st.selectbox("Vehicle Class", list(le_vehicle.classes_))
fuel_type     = st.selectbox("Fuel Type", list(le_fuel.classes_))
transmission  = st.selectbox("Transmission", list(le_trans.classes_))
cylinders     = st.selectbox("Cylinders", [3, 4, 5, 6, 8, 10, 12, 16])
engine_size   = st.slider("Engine Size (L)", 1.0, 8.0, 2.0, step=0.1)
fuel_comb     = st.slider("Fuel Consumption Comb (L/100 km)", 4.0, 25.0, 9.5, step=0.1)

# Predict Button
if st.button("Predict CO2 Emission"):
    # Encode categorical inputs
    vc_encoded = le_vehicle.transform([vehicle_class])[0]
    ft_encoded = le_fuel.transform([fuel_type])[0]
    tr_encoded = le_trans.transform([transmission])[0]

    # Create dataframe with raw values
    sample = pd.DataFrame([[vc_encoded, ft_encoded, tr_encoded,
                            cylinders, engine_size, fuel_comb]],
                          columns=['Vehicle Class', 'Fuel Type', 'Transmission',
                                   'Cylinders', 'Engine Size(L)',
                                   'Fuel Consumption Comb (L/100 km)'])

    # Scale ONLY Engine Size and Fuel Consumption (same as training)
    cols = ['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']
    sample[cols] = scaler.transform(sample[cols])

    prediction = model.predict(sample)
    st.success(f"🌿 Predicted CO2 Emission: **{round(float(prediction[0]), 2)} g/km**")