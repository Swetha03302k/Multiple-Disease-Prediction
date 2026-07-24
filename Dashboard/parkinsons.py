import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("../Models/parkinsons_model.pkl")
scaler = joblib.load("../Models/parkinsons_scaler.pkl")


def app():

    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("---")

    st.subheader("🎤 Voice Measurement Parameters")

    col1, col2 = st.columns(2)

    with col1:

        fo = st.number_input("MDVP:Fo (Hz)", value=120.0)
        fhi = st.number_input("MDVP:Fhi (Hz)", value=140.0)
        flo = st.number_input("MDVP:Flo (Hz)", value=110.0)
        jitter_percent = st.number_input("MDVP:Jitter (%)", value=0.005)
        jitter_abs = st.number_input("MDVP:Jitter (Abs)", value=0.00005)
        rap = st.number_input("MDVP:RAP", value=0.003)
        ppq = st.number_input("MDVP:PPQ", value=0.003)
        ddp = st.number_input("Jitter:DDP", value=0.009)
        shimmer = st.number_input("MDVP:Shimmer", value=0.03)
        shimmer_db = st.number_input("MDVP:Shimmer (dB)", value=0.30)
        apq3 = st.number_input("Shimmer:APQ3", value=0.015)

    with col2:

        apq5 = st.number_input("Shimmer:APQ5", value=0.020)
        apq = st.number_input("MDVP:APQ", value=0.025)
        dda = st.number_input("Shimmer:DDA", value=0.045)
        nhr = st.number_input("NHR", value=0.020)
        hnr = st.number_input("HNR", value=21.0)
        rpde = st.number_input("RPDE", value=0.45)
        dfa = st.number_input("DFA", value=0.72)
        spread1 = st.number_input("Spread1", value=-5.0)
        spread2 = st.number_input("Spread2", value=0.25)
        d2 = st.number_input("D2", value=2.3)
        ppe = st.number_input("PPE", value=0.20)

    st.markdown("---")

    predict = st.button("🔍 Predict Parkinson's Disease")

    if predict:

        input_data = np.array([[
            fo,
            fhi,
            flo,
            jitter_percent,
            jitter_abs,
            rap,
            ppq,
            ddp,
            shimmer,
            shimmer_db,
            apq3,
            apq5,
            apq,
            dda,
            nhr,
            hnr,
            rpde,
            dfa,
            spread1,
            spread2,
            d2,
            ppe
        ]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)

        # Result
        if prediction[0] == 1:

            st.error(
                "⚠️ Prediction: The patient is likely to have Parkinson's Disease."
            )

            st.write(
                "This prediction indicates that the entered voice measurements "
                "are consistent with patterns associated with Parkinson's Disease. "
                "Please consult a neurologist or qualified healthcare professional "
                "for a comprehensive medical evaluation and confirmatory diagnostic tests."
            )

        else:

            st.success(
                "✅ Prediction: The patient is unlikely to have Parkinson's Disease."
            )

            st.write(
                "Based on the entered voice measurements, the model predicts a "
                "lower likelihood of Parkinson's Disease. However, this result "
                "should not be considered a medical diagnosis. If symptoms persist "
                "or you have health concerns, please consult a qualified healthcare "
                "professional."
            )