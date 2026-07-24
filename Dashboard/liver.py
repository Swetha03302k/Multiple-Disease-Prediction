import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("../Models/liver_model.pkl")
scaler = joblib.load("../Models/liver_scaler.pkl")


def app():

    st.title("🫀 Liver Disease Prediction")
    st.markdown("---")

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        total_bilirubin = st.number_input(
            "Total Bilirubin",
            value=1.0
        )

        direct_bilirubin = st.number_input(
            "Direct Bilirubin",
            value=0.3
        )

        alkaline_phosphotase = st.number_input(
            "Alkaline Phosphotase",
            value=200.0
        )

    with col2:

        alamine_aminotransferase = st.number_input(
            "Alamine Aminotransferase",
            value=30.0
        )

        aspartate_aminotransferase = st.number_input(
            "Aspartate Aminotransferase",
            value=30.0
        )

        total_proteins = st.number_input(
            "Total Proteins",
            value=6.5
        )

        albumin = st.number_input(
            "Albumin",
            value=3.5
        )

        albumin_globulin_ratio = st.number_input(
            "Albumin & Globulin Ratio",
            value=1.0
        )

    st.markdown("---")

    predict = st.button("🔍 Predict Liver Disease")

    if predict:

        # Encode Gender
        gender = 1 if gender == "Male" else 0

        # Create input array
        input_data = np.array([[
            age,
            gender,
            total_bilirubin,
            direct_bilirubin,
            alkaline_phosphotase,
            alamine_aminotransferase,
            aspartate_aminotransferase,
            total_proteins,
            albumin,
            albumin_globulin_ratio
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)

        # Result
        if prediction[0] == 1:

            st.error(
                "⚠️ Prediction: The patient is likely to have Liver Disease."
            )

            st.write(
                "This prediction indicates that the entered clinical values "
                "are consistent with patterns associated with Liver Disease. "
                "Please consult a qualified healthcare professional for a "
                "comprehensive medical evaluation and appropriate diagnostic tests."
            )

        else:

            st.success(
                "✅ Prediction: The patient is unlikely to have Liver Disease."
            )

            st.write(
                "Based on the entered clinical values, the model predicts a "
                "lower likelihood of Liver Disease. However, this result "
                "should not be considered a medical diagnosis. If symptoms "
                "persist or you have health concerns, please consult a "
                "qualified healthcare professional."
            )