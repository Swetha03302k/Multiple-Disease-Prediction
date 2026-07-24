import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load("../Models/kidney_model.pkl")
scaler = joblib.load("../Models/kidney_scaler.pkl")


def app():

    st.title("🩺 Kidney Disease Prediction")
    st.markdown("---")

    st.subheader("👤 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=30)
        bp = st.number_input("Blood Pressure", value=80.0)
        sg = st.number_input("Specific Gravity", value=1.020)
        al = st.number_input("Albumin", value=0.0)
        su = st.number_input("Sugar", value=0.0)
        bgr = st.number_input("Blood Glucose Random", value=120.0)
        bu = st.number_input("Blood Urea", value=30.0)

    with col2:
        sc = st.number_input("Serum Creatinine", value=1.2)
        sod = st.number_input("Sodium", value=135.0)
        pot = st.number_input("Potassium", value=4.5)
        hemo = st.number_input("Hemoglobin", value=15.0)
        pcv = st.number_input("Packed Cell Volume", value=45.0)
        wc = st.number_input("White Blood Cell Count", value=8000.0)
        rc = st.number_input("Red Blood Cell Count", value=5.0)

    st.markdown("---")

    st.subheader("🩺 Clinical Information")

    col3, col4 = st.columns(2)

    with col3:
        rbc = st.selectbox(
            "Red Blood Cells",
            ["normal", "abnormal"]
        )

        pc = st.selectbox(
            "Pus Cell",
            ["normal", "abnormal"]
        )

        pcc = st.selectbox(
            "Pus Cell Clumps",
            ["present", "notpresent"]
        )

        ba = st.selectbox(
            "Bacteria",
            ["present", "notpresent"]
        )

        htn = st.selectbox(
            "Hypertension",
            ["yes", "no"]
        )

    with col4:
        dm = st.selectbox(
            "Diabetes Mellitus",
            ["yes", "no"]
        )

        cad = st.selectbox(
            "Coronary Artery Disease",
            ["yes", "no"]
        )

        appet = st.selectbox(
            "Appetite",
            ["good", "poor"]
        )

        pe = st.selectbox(
            "Pedal Edema",
            ["yes", "no"]
        )

        ane = st.selectbox(
            "Anemia",
            ["yes", "no"]
        )

    st.markdown("---")

    predict = st.button("🔍 Predict Kidney Disease")

    if predict:

        # Encode categorical inputs
        rbc = 1 if rbc == "normal" else 0
        pc = 1 if pc == "normal" else 0
        pcc = 1 if pcc == "present" else 0
        ba = 1 if ba == "present" else 0
        htn = 1 if htn == "yes" else 0
        dm = 1 if dm == "yes" else 0
        cad = 1 if cad == "yes" else 0
        appet = 1 if appet == "good" else 0
        pe = 1 if pe == "yes" else 0
        ane = 1 if ane == "yes" else 0

        # Create input array
        input_data = np.array([[
            age, bp, sg, al, su,
            rbc, pc, pcc, ba,
            bgr, bu, sc, sod, pot,
            hemo, pcv, wc, rc,
            htn, dm, cad, appet,
            pe, ane
        ]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)

        # Display result
        if prediction[0] == 1:
            st.error("⚠️ Prediction: The patient is likely to have Chronic Kidney Disease.")

            st.write(
                "This prediction indicates that the entered clinical values are consistent "
                "with patterns associated with Chronic Kidney Disease (CKD). "
                "Please consult a qualified healthcare professional for a comprehensive "
                "medical evaluation and confirmatory diagnostic tests."
            )

        else:
            st.success("✅ Prediction: The patient is unlikely to have Chronic Kidney Disease.")

            st.write(
                "Based on the entered clinical values, the model predicts a lower likelihood "
                "of Chronic Kidney Disease (CKD). However, this result should not be "
                "considered a medical diagnosis. If symptoms persist or you have health "
                "concerns, consult a healthcare professional."
            )