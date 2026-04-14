import streamlit as st

from app import (
    FEATURE_COLUMNS,
    fetch_live_weather,
    get_district_map,
    get_history,
    get_location_features,
    get_location_percentages,
    get_locations,
    get_previous_history_counts,
    get_safety_guidance,
    parse_manual_features,
    predict_probabilities,
    estimate_time_to_occurrence,
)

st.set_page_config(
    page_title="Disaster Predictor",
    page_icon="🌧️",
    layout="wide",
)

st.title("Disaster Predictor")
st.markdown(
    "Predict Flood, Wildfire, and Earthquake risk for Indian states and districts using live weather and local historical data."
)


@st.cache_data(show_spinner=False)
def load_locations():
    return get_locations()


@st.cache_data(show_spinner=False)
def load_district_map():
    return get_district_map()


@st.cache_data(show_spinner=False)
def get_live_weather(state: str, district: str):
    return fetch_live_weather(state, district)


locations = load_locations()
district_map = load_district_map()

with st.sidebar:
    st.header("Your location")
    selected_state = st.selectbox("State", [""] + locations)
    district_options = district_map.get(selected_state, []) if selected_state else []
    selected_district = st.selectbox("District", [""] + district_options)

    st.markdown("---")
    st.header("Weather values (optional)")
    temperature = st.text_input("Temperature (°C)", "")
    rainfall = st.text_input("Rainfall (mm)", "")
    humidity = st.text_input("Humidity (%)", "")
    wind_speed = st.text_input("Wind Speed (km/h)", "")
    soil_moisture = st.text_input("Soil Moisture (%)", "")

    submitted = st.button("Predict risk")

if not submitted:
    st.info(
        "Select a state and district from the sidebar, optionally enter weather values, then click **Predict risk**."
    )
    st.write("The app falls back to live weather or average climate values when your weather values are not provided.")

if submitted:
    if not selected_state or not selected_district:
        st.error("Please select both a state and a district before predicting.")
    else:
        form_inputs = {
            "temperature": temperature,
            "rainfall": rainfall,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "soil_moisture": soil_moisture,
        }
        manual_features = parse_manual_features(form_inputs)
        weather_source = "manual"
        weather_report = None

        if manual_features is None:
            live_weather = get_live_weather(selected_state, selected_district)
            if live_weather is not None:
                manual_features = live_weather["features"]
                weather_report = live_weather["report"]
                weather_source = "live"
            else:
                manual_features = get_location_features(selected_state, selected_district)
                weather_source = "average"

        if manual_features is None:
            st.warning("Weather values could not be determined for this location.")
        else:
            features = manual_features
            percentages = get_location_percentages(selected_state, selected_district)
            history = get_history(selected_state, selected_district)
            history_counts = get_previous_history_counts(selected_state, selected_district)
            predictions = predict_probabilities(features, percentages, history_counts)
            time_estimates = estimate_time_to_occurrence(predictions, percentages)
            top_prediction = max(predictions, key=predictions.get) if predictions else None
            safety_guidance = get_safety_guidance(top_prediction)

            st.success(f"Prediction complete for {selected_district}, {selected_state}.")
            st.markdown(f"**Weather source:** {weather_source.capitalize()}")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Weather data used")
                st.write(
                    f"""Temperature: {features[0]} °C  
Rainfall: {features[1]} mm  
Humidity: {features[2]} %  
Wind speed: {features[3]} km/h  
Soil moisture: {features[4]} %"""
                )
                if weather_report:
                    st.write("---")
                    st.write("**Live weather report**")
                    st.write(
                        f"""Daily high: {weather_report.get('daily_max')} °C  
Daily low: {weather_report.get('daily_min')} °C  
Daily precipitation: {weather_report.get('daily_precipitation')} mm  
Daily wind max: {weather_report.get('daily_wind_max')} km/h"""
                    )

            with col2:
                st.subheader("Top prediction")
                if top_prediction:
                    st.metric("Leading hazard", top_prediction, f"{predictions[top_prediction]}% chance")
                else:
                    st.write("Prediction model could not determine a top hazard.")

            st.markdown("---")
            st.subheader("Risk percentages")
            risk_data = {
                "Disaster": list(predictions.keys()),
                "Predicted chance (%)": list(predictions.values()),
            }
            st.table(risk_data)

            st.subheader("Historical occurrence")
            history_data = {
                "Disaster": list(percentages.keys()),
                "Historical percentage (%)": list(percentages.values()),
            }
            st.table(history_data)

            st.subheader("Estimated timing")
            timing_data = {
                "Disaster": list(time_estimates.keys()),
                "Expected window": list(time_estimates.values()),
            }
            st.table(timing_data)

            st.markdown("---")
            st.subheader(safety_guidance["title"])
            col3, col4 = st.columns(2)
            with col3:
                st.write("**Before the disaster**")
                for item in safety_guidance["before"]:
                    st.write(f"- {item}")
            with col4:
                st.write("**During the disaster**")
                for item in safety_guidance["during"]:
                    st.write(f"- {item}")

            st.markdown("---")
            st.subheader("Historical disaster records")
            if history:
                history_table = {
                    "Year": [row["year"] for row in history],
                    "Disaster": [row["disaster_type"] for row in history],
                    "Records": [row["count"] for row in history],
                }
                st.table(history_table)
            else:
                st.info("No historical disaster records are available for this location.")
