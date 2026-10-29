import streamlit as st

from src.weather_predictor import predict_weather

st.title("Weather Forecasting ML")

st.write("Enter a city to predict tomorrow's weather.")


location = st.text_input("Enter Location", placeholder="Example: Pune")


if st.button("Predict Weather"):

    if not location.strip():
        st.warning("Please enter a location.")

    else:
        with st.spinner("Getting weather prediction..."):

            result = predict_weather(location.strip())

        if result is None:
            st.error("Location not found. Please try another city.")

        else:
            st.success("Prediction completed successfully!")

            st.subheader("Prediction Result")

            st.write("**Location:**", result["location"])

            st.write("**Predicted weather for tomorrow:**", result["prediction"])

            st.write("**Prediction confidence:**", f'{result["confidence"]}%')
