import os
import sqlite3
import pickle
import requests
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "disaster.db"
MODEL_PATH = BASE_DIR / "model.pkl"
APP_URL = os.environ.get("APP_URL", "https://disaster-predictor.onrender.com")
FEATURE_COLUMNS = [
    "temperature",
    "rainfall",
    "humidity",
    "wind_speed",
    "soil_moisture",
]

DISTRICT_MAP = {
    "Andhra Pradesh": [
        "Visakhapatnam",
        "Vijayawada",
        "Tirupati",
        "Guntur",
        "Nellore",
        "Anantapur",
        "Chittoor",
        "Krishna",
        "Kurnool",
        "Prakasam",
        "Srikakulam",
        "Vizianagaram",
        "West Godavari",
        "East Godavari",
        "Eluru",
        "Nandyal",
        "Palnadu",
        "YSR Kadapa",
        "NTR",
        "Kakinada",
        "Alluri Sitharama Raju",
        "Anakapalli",
        "Annamayya",
        "Ambedkar Konaseema",
        "Parvathipuram Manyam",
    ],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Tezu"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat", "Tezpur"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bhilai", "Jagdalpur"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Haryana": ["Chandigarh", "Gurugram", "Panipat", "Faridabad", "Ambala"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Manali", "Kullu", "Solan"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Dumka", "Bokaro"],
    "Karnataka": ["Bengaluru", "Mysore", "Mangalore", "Hubli", "Belgaum"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kannur"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Manipur": ["Imphal", "Thoubal", "Ukhrul"],
    "Meghalaya": ["Shillong", "Tura", "Jowai"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri", "Berhampur"],
    "Punjab": ["Amritsar", "Ludhiana", "Jalandhar", "Patiala", "Bathinda"],
    "Rajasthan": ["Jaipur", "Udaipur", "Jodhpur", "Jaisalmer", "Bikaner"],
    "Sikkim": ["Gangtok", "Namchi", "Geyzing"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli", "Salem"],
    "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar", "Khammam"],
    "Tripura": ["Agartala", "Udaipur", "Dharmanagar"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Noida"],
    "Uttarakhand": ["Dehradun", "Nainital", "Haridwar", "Haldwani", "Rudrapur"],
    "West Bengal": ["Kolkata", "Darjeeling", "Asansol", "Durgapur", "Siliguri"],
}

DISTRICT_COORDINATES = {
    "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567),
    "Nagpur": (21.1458, 79.0882),
    "Lucknow": (26.8467, 80.9462),
    "Kanpur": (26.4499, 80.3319),
    "Varanasi": (25.3176, 82.9739),
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Bengaluru": (12.9716, 77.5946),
    "Mysore": (12.2958, 76.6394),
    "Mangalore": (12.9141, 74.8560),
    "Ahmedabad": (23.0225, 72.5714),
    "Surat": (21.1702, 72.8311),
    "Vadodara": (22.3072, 73.1812),
    "Kolkata": (22.5726, 88.3639),
    "Darjeeling": (27.0360, 88.2627),
    "Asansol": (23.6848, 86.9626),
    "Jaipur": (26.9124, 75.7873),
    "Udaipur": (24.5854, 73.7125),
    "Jodhpur": (26.2389, 73.0243),
    "Patna": (25.5941, 85.1376),
    "Gaya": (24.7954, 85.0002),
    "Bhagalpur": (25.2441, 86.9842),
    "Visakhapatnam": (17.6868, 83.2185),
    "Vijayawada": (16.5062, 80.6480),
    "Tirupati": (13.6288, 79.4192),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kochi": (9.9312, 76.2673),
    "Kozhikode": (11.2588, 75.7804),
    "Bhopal": (23.2599, 77.4126),
    "Indore": (22.7196, 75.8577),
    "Gwalior": (26.2183, 78.1828),
    "Bhubaneswar": (20.2961, 85.8245),
    "Cuttack": (20.4625, 85.8828),
    "Rourkela": (22.2604, 84.8536),
    "Chandigarh": (30.7333, 76.7794),
    "Gurugram": (28.4595, 77.0266),
    "Panipat": (29.3909, 76.9635),
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Jalandhar": (31.3260, 75.5762),
    "Hyderabad": (17.3850, 78.4867),
    "Warangal": (17.9689, 79.5941),
    "Nizamabad": (18.6726, 78.0941),
    "Guwahati": (26.1445, 91.7362),
    "Dibrugarh": (27.4728, 94.9110),
    "Silchar": (24.8333, 92.7789),
    "Ranchi": (23.3441, 85.3096),
    "Jamshedpur": (22.8046, 86.2029),
    "Dhanbad": (23.7957, 86.4304),
    "Raipur": (21.2514, 81.6296),
    "Bilaspur": (22.0796, 82.1391),
    "Durg": (21.1907, 81.2840),
    "Dehradun": (30.3165, 78.0322),
    "Nainital": (29.3919, 79.4542),
    "Haridwar": (29.9457, 78.1642),
    "Shimla": (31.1048, 77.1734),
    "Dharamshala": (32.2190, 76.3234),
    "Manali": (32.2396, 77.1887),
    "Guntur": (16.3067, 80.4365),
    "Nellore": (14.4426, 79.9865),
    "Itanagar": (27.0844, 93.6053),
    "Tawang": (27.5860, 91.8650),
    "Tezu": (27.8847, 96.1520),
    "Jorhat": (26.7440, 94.2038),
    "Tezpur": (26.6333, 92.8036),
    "Muzaffarpur": (26.1200, 85.3830),
    "Darbhanga": (26.1542, 85.8970),
    "Bhilai": (21.1957, 81.2930),
    "Jagdalpur": (19.0748, 82.0287),
    "Panaji": (15.4909, 73.8278),
    "Margao": (15.2867, 73.9578),
    "Vasco da Gama": (15.3970, 73.8176),
    "Rajkot": (22.3039, 70.8022),
    "Bhavnagar": (21.7645, 72.1519),
    "Faridabad": (28.4089, 77.3178),
    "Ambala": (30.3782, 76.7767),
    "Kullu": (31.9645, 77.1091),
    "Solan": (30.9082, 77.0968),
    "Dumka": (24.2674, 87.2497),
    "Bokaro": (23.6693, 86.1511),
    "Hubli": (15.3647, 75.1240),
    "Belgaum": (15.8497, 74.4977),
    "Thrissur": (10.5276, 76.2144),
    "Kannur": (11.8745, 75.3704),
    "Jabalpur": (23.1815, 79.9864),
    "Ujjain": (23.1793, 75.7849),
    "Nashik": (19.9975, 73.7898),
    "Aurangabad": (19.8762, 75.3433),
    "Imphal": (24.8170, 93.9368),
    "Thoubal": (24.6083, 93.9665),
    "Ukhrul": (25.1150, 94.3639),
    "Shillong": (25.5788, 91.8933),
    "Tura": (25.5100, 90.2167),
    "Jowai": (25.4584, 92.2588),
    "Aizawl": (23.7271, 92.7176),
    "Lunglei": (22.9171, 92.7796),
    "Champhai": (23.9100, 93.2650),
    "Kohima": (25.6740, 94.1106),
    "Dimapur": (25.9045, 93.7475),
    "Mokokchung": (26.3250, 94.5231),
    "Puri": (19.8135, 85.8312),
    "Berhampur": (19.3148, 84.7941),
    "Patiala": (30.3398, 76.3869),
    "Bathinda": (30.2110, 74.9455),
    "Jaisalmer": (26.9157, 70.9083),
    "Bikaner": (28.0180, 73.3119),
    "Gangtok": (27.3314, 88.6130),
    "Namchi": (27.1660, 88.5036),
    "Geyzing": (27.3000, 88.3500),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Karimnagar": (18.4386, 79.1288),
    "Khammam": (17.2473, 80.1514),
    "Agartala": (23.8315, 91.2868),
    "Tripura|Udaipur": (23.5500, 91.5000),
    "Dharmanagar": (24.0000, 92.5000),
    "New Delhi": (28.6139, 77.2090),
    "Srinagar": (34.0837, 74.7973),
    "Jammu": (32.7266, 74.8570),
    "Anantnag": (33.7300, 75.1500),
    "Leh": (34.1526, 77.5770),
    "Kargil": (34.5553, 76.1334),
    "Port Blair": (11.6234, 92.7265),
    "Silvassa": (20.2710, 73.0035),
    "Daman": (20.4050, 72.8500),
    "Puducherry": (11.9416, 79.8083),
    "Karaikal": (10.9252, 79.8387),
    "Kavaratti": (10.5663, 72.6417),
}

STATE_COORDINATES = {
    "Andaman & Nicobar": (11.7401, 92.6586),
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (28.2180, 94.7278),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chandigarh": (30.7333, 76.7794),
    "Chhattisgarh": (21.2787, 81.8661),
    "Dadra & Nagar Haveli & Daman & Diu": (20.1809, 73.0169),
    "Delhi": (28.7041, 77.1025),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jammu & Kashmir": (33.7782, 76.5762),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Ladakh": (34.1526, 77.5770),
    "Lakshadweep": (10.3280, 72.7847),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Puducherry": (11.9416, 79.8083),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
}


def get_coordinates(state, district):
    state_key = f"{state}|{district}"
    return DISTRICT_COORDINATES.get(state_key) or DISTRICT_COORDINATES.get(district) or STATE_COORDINATES.get(state)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")


def get_db_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database file not found. Run init_db.py first to create {DB_PATH}."
        )

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found. Run init_db.py first to create {MODEL_PATH}."
        )

    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)

MODEL_DATA = load_model()
MODEL = MODEL_DATA["model"]
MODEL_CLASSES = MODEL_DATA["classes"]


def get_locations():
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT state FROM state_stats ORDER BY state"
        ).fetchall()
        return [row["state"] for row in rows]


def get_district_map():
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT state, district FROM district_stats ORDER BY state, district"
        ).fetchall()
        district_map = {}
        for row in rows:
            district_map.setdefault(row["state"], []).append(row["district"])

    for state, fallback_districts in DISTRICT_MAP.items():
        if state not in district_map:
            district_map[state] = fallback_districts.copy()
        else:
            for district in fallback_districts:
                if district not in district_map[state]:
                    district_map[state].append(district)

    return district_map


def get_districts(state):
    return get_district_map().get(state, [])


def get_location_percentages(state, district):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT disaster_type, percentage FROM district_stats WHERE state = ? AND district = ? ORDER BY disaster_type",
            (state, district),
        ).fetchall()
        if rows:
            return {row["disaster_type"]: round(row["percentage"], 1) for row in rows}

        rows = conn.execute(
            "SELECT disaster_type, percentage FROM state_stats WHERE state = ? ORDER BY disaster_type",
            (state,),
        ).fetchall()
        if rows:
            return {row["disaster_type"]: round(row["percentage"], 1) for row in rows}

        rows = conn.execute(
            "SELECT disaster_type, percentage FROM global_stats ORDER BY disaster_type"
        ).fetchall()
        return {row["disaster_type"]: round(row["percentage"], 1) for row in rows}


def get_location_features(state, district):
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT temperature, rainfall, humidity, wind_speed, soil_moisture FROM district_features WHERE state = ? AND district = ?",
            (state, district),
        ).fetchone()
        if row is not None:
            return [row["temperature"], row["rainfall"], row["humidity"], row["wind_speed"], row["soil_moisture"]]

        row = conn.execute(
            "SELECT temperature, rainfall, humidity, wind_speed, soil_moisture FROM state_features WHERE state = ?",
            (state,),
        ).fetchone()
        if row is None:
            return None

        return [row["temperature"], row["rainfall"], row["humidity"], row["wind_speed"], row["soil_moisture"]]


def get_history(state, district):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT year, disaster_type, record_count FROM history_summary WHERE state = ? AND district = ? ORDER BY year DESC",
            (state, district),
        ).fetchall()
        if rows:
            return [
                {
                    "year": row["year"],
                    "disaster_type": row["disaster_type"],
                    "count": row["record_count"],
                }
                for row in rows
            ]

        rows = conn.execute(
            "SELECT year, disaster_type, record_count FROM state_history WHERE state = ? ORDER BY year DESC",
            (state,),
        ).fetchall()
        return [
            {
                "year": row["year"],
                "disaster_type": row["disaster_type"],
                "count": row["record_count"],
            }
            for row in rows
        ]


def get_previous_history_counts(state, district):
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT disaster_type, SUM(record_count) AS total_records FROM history_summary WHERE state = ? AND district = ? GROUP BY disaster_type",
            (state, district),
        ).fetchall()
        if rows:
            return {row["disaster_type"]: row["total_records"] for row in rows}

        rows = conn.execute(
            "SELECT disaster_type, SUM(record_count) AS total_records FROM state_history WHERE state = ? GROUP BY disaster_type",
            (state,),
        ).fetchall()
        return {row["disaster_type"]: row["total_records"] for row in rows}


def estimate_time_to_occurrence(predictions, percentages):
    if not predictions:
        return {}

    time_estimates = {}
    for disaster, probability in predictions.items():
        history_pct = percentages.get(disaster, 0)
        score = probability * 0.7 + history_pct * 0.3
        if score >= 70:
            window = "0-24 hours"
        elif score >= 50:
            window = "1-3 days"
        elif score >= 30:
            window = "4-10 days"
        else:
            window = "More than 10 days"
        time_estimates[disaster] = window
    return time_estimates


def get_safety_guidance(disaster_type):
    guidance = {
        "Flood": {
            "title": "Flood preparedness and response",
            "before": [
                "Keep emergency supplies ready, including water, non-perishable food, and medicine.",
                "Know evacuation routes and high-ground shelters near your district.",
                "Protect important documents in waterproof containers and unplug electrical appliances.",
                "Monitor flood alerts from local authorities and avoid driving through water-covered roads.",
            ],
            "during": [
                "Move to higher ground immediately if water levels rise or evacuation orders are issued.",
                "Avoid wading through floodwater; it can conceal debris, strong currents, or electrical hazards.",
                "Stay updated on official river and weather alerts until waters recede.",
                "If trapped, climb to the roof or higher floor and signal for help.",
            ],
        },
        "Wildfire": {
            "title": "Wildfire safety and evacuation",
            "before": [
                "Create a wildfire action plan and identify multiple escape routes from your district.",
                "Clear dry leaves, brush, and combustible materials from around your home.",
                "Keep windows and vents closed and store masks and a first aid kit nearby.",
                "Follow local fire warnings and be ready to leave immediately if smoke or flame approaches.",
            ],
            "during": [
                "Evacuate immediately when advised; do not wait for flames to arrive.",
                "Wear a mask or damp cloth over your face to reduce smoke inhalation.",
                "If trapped, shelter inside a building away from outside walls and stay low to the floor.",
                "Keep listening to emergency broadcasts for the latest wildfire updates and routes.",
            ],
        },
        "Earthquake": {
            "title": "Earthquake readiness and sheltering",
            "before": [
                "Secure heavy furniture and glass items so they do not fall during shaking.",
                "Keep an emergency kit with water, food, flashlight, and a mobile charger ready.",
                "Practice 'Drop, Cover, and Hold On' with family members for quick response.",
                "Identify safe spots inside your home, such as under sturdy tables or against interior walls.",
            ],
            "during": [
                "Drop to the ground, cover your head and neck, and hold on until shaking stops.",
                "Stay away from windows, tall furniture, and exterior walls.",
                "If outdoors, move to an open area away from buildings, power lines, and trees.",
                "After shaking stops, check for hazards and listen to local authorities for safety instructions.",
            ],
        },
    }
    return guidance.get(disaster_type, {
        "title": "General disaster preparedness",
        "before": [
            "Stay informed about weather and emergency alerts for your area.",
            "Prepare a basic emergency kit with water, food, medicines, and important documents.",
            "Have a communication plan with family members and a safe meeting place.",
        ],
        "during": [
            "Follow instructions from local authorities and emergency responders.",
            "Stay calm and help those around you if it is safe to do so.",
            "Move to a safe location and avoid unnecessary travel during the event.",
        ],
    })


def build_feature_cards(state, district, predictions, percentages, history, weather_source, features):
    top_prediction = None
    if predictions:
        top_prediction = max(predictions.items(), key=lambda item: item[1])

    highest_historical = None
    if percentages:
        highest_historical = max(percentages.items(), key=lambda item: item[1])

    history_text = "No historical records found for this location."
    if history:
        latest = history[0]
        history_text = (
            f"{latest['year']} {latest['disaster_type']} — {latest['count']} records"
        )

    if weather_source == "live":
        forecast_text = "Live forecast data was fetched for this location."
    elif weather_source == "manual":
        forecast_text = "Weather values were entered manually for the prediction."
    else:
        forecast_text = "Average location climate data was used because live weather was unavailable."

    feature_text = "Weather data unavailable."
    if features:
        feature_text = (
            f"Temperature {features[0]}°C · Rainfall {features[1]}mm · "
            f"Humidity {features[2]}% · Wind {features[3]} km/h · Soil {features[4]}%"
        )

    adaptive_text = (
        f"Theme and risk visuals are tuned for {district}, {state} "
        "using local historical patterns and current weather data."
    )

    smart_text = (
        f"Top prediction: {top_prediction[0]} with {top_prediction[1]}%."
        if top_prediction
        else "Smart prediction could not be determined."
    )

    risk_text = (
        f"Top mapped risk: {highest_historical[0]} at {highest_historical[1]}%."
        if highest_historical
        else "Risk mapping data unavailable."
    )

    ai_text = (
        f"AI predicts {top_prediction[0]} as the leading hazard with {top_prediction[1]}%."
        if top_prediction
        else "AI prediction is not available."
    )

    return [
        {
            "title": "Adaptive Theme",
            "description": adaptive_text,
            "icon_class": "icon-adaptive",
            "icon_svg": (
                '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<circle cx="32" cy="32" r="14" stroke="currentColor" stroke-width="3" />'
                '<path d="M17 32h30" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '<path d="M32 18c7 6 7 14 0 20" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '</svg>'
            ),
        },
        {
            "title": "History Insight",
            "description": history_text,
            "icon_class": "icon-history",
            "icon_svg": (
                '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<circle cx="32" cy="32" r="18" stroke="currentColor" stroke-width="3" />'
                '<path d="M32 20v12l8 8" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '<path d="M32 10a22 22 0 1 1-15 6" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '</svg>'
            ),
        },
        {
            "title": "Smart Results",
            "description": smart_text,
            "icon_class": "icon-smart",
            "icon_svg": (
                '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M24 26c0-5.5 4.5-10 10-10s10 4.5 10 10c0 4-2.5 7.5-5 9v7H29v-7c-2.5-1.5-5-5-5-9Z" stroke="currentColor" stroke-width="3" />'
                '<path d="M29 41h6" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '<path d="M31 45h2" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '</svg>'
            ),
        },
        {
            "title": "Live Forecast",
            "description": f"{forecast_text} {feature_text}",
            "icon_class": "icon-live",
            "icon_svg": (
                '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M24 34a10 10 0 0 1 20 0h2a8 8 0 1 1 0 16H24a8 8 0 1 1 0-16Z" stroke="currentColor" stroke-width="3" />'
                '<path d="M30 46l-4 8M38 46l-4 8" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '</svg>'
            ),
        },
        {
            "title": "Risk Mapping",
            "description": risk_text,
            "icon_class": "icon-risk",
            "icon_svg": (
                '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M32 10l18 8v12c0 14-10 24-18 24S14 44 14 30V18l18-8Z" stroke="currentColor" stroke-width="3" />'
                '<path d="M32 24v10M32 38h0" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '</svg>'
            ),
        },
        {
            "title": "AI Prediction",
            "description": ai_text,
            "icon_class": "icon-ai",
            "icon_svg": (
                '<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">'
                '<path d="M22 22h20M22 30h20M22 38h20" stroke="currentColor" stroke-width="3" stroke-linecap="round" />'
                '<circle cx="42" cy="22" r="3" fill="currentColor" />'
                '<circle cx="42" cy="38" r="3" fill="currentColor" />'
                '<circle cx="22" cy="46" r="3" fill="currentColor" />'
                '</svg>'
            ),
        },
    ]


def parse_manual_features(form):
    values = []
    for name in FEATURE_COLUMNS:
        raw = form.get(name)
        if raw is None or raw.strip() == "":
            return None
        try:
            values.append(float(raw))
        except ValueError:
            return None
    return values


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def predict_probabilities(features, percentages=None, history_counts=None):
    if not features:
        return {}

    temperature, rainfall, humidity, wind_speed, soil_moisture = features
    rain_norm = clamp((rainfall - 5) / 100)
    temp_norm = clamp((temperature - 20) / 25)
    humidity_norm = clamp(humidity / 100)
    soil_norm = clamp(soil_moisture / 100)
    wind_norm = clamp(wind_speed / 120)

    weather_scores = {
        "Flood": clamp(0.5 * rain_norm + 0.3 * humidity_norm + 0.2 * soil_norm),
        "Wildfire": clamp(0.35 * temp_norm + 0.35 * (1 - humidity_norm) + 0.2 * (1 - soil_norm) + 0.1 * wind_norm),
        "Earthquake": 0.2,
    }

    percentages = percentages or {}
    history_counts = history_counts or {}
    total_history = sum(history_counts.values())
    estimates = {}
    for disaster in MODEL_CLASSES:
        base = clamp((percentages.get(disaster, 15) / 100) * 0.8 + 0.2)
        weather_value = weather_scores.get(disaster, 0.25)
        history_norm = clamp((history_counts.get(disaster, 0) / total_history) if total_history else 0.2)
        raw_score = base * 0.5 + weather_value * 0.3 + history_norm * 0.2
        if disaster == "Earthquake":
            raw_score = clamp(0.13 + base * 0.55 + history_norm * 0.12)
        estimates[disaster] = round(raw_score * 100, 1)

    total = sum(estimates.values())
    if total == 0:
        return {disaster: 0.0 for disaster in MODEL_CLASSES}

    normalized = {}
    for disaster, score in estimates.items():
        normalized[disaster] = round(score / total * 100, 1)
    return normalized


def fetch_live_weather(state, district):
    coords = get_coordinates(state, district)
    if coords is None:
        return None

    latitude, longitude = coords
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": "true",
        "hourly": "relativehumidity_2m,soil_moisture_0_1cm,precipitation",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "auto",
    }

    try:
        response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    current = data.get("current_weather", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    time_values = hourly.get("time", [])
    daily_times = daily.get("time", [])
    if not time_values or not daily_times:
        return None

    latest_index = len(time_values) - 1
    temperature = current.get("temperature")
    wind_speed = current.get("windspeed")
    humidity = hourly.get("relativehumidity_2m", [None])[latest_index]
    rainfall = hourly.get("precipitation", [None])[latest_index]
    soil_moisture = hourly.get("soil_moisture_0_1cm", [None])[latest_index]
    daily_max = daily.get("temperature_2m_max", [None])[0]
    daily_min = daily.get("temperature_2m_min", [None])[0]
    daily_precip = daily.get("precipitation_sum", [None])[0]
    daily_wind = daily.get("windspeed_10m_max", [None])[0]

    if any(value is None for value in [temperature, humidity, rainfall, wind_speed, soil_moisture]):
        return None

    features = [
        float(temperature),
        float(rainfall),
        float(humidity),
        float(wind_speed),
        float(soil_moisture) * 100,
    ]

    weather_report = {
        "temperature": float(temperature),
        "rainfall": float(rainfall),
        "humidity": float(humidity),
        "wind_speed": float(wind_speed),
        "soil_moisture": float(soil_moisture) * 100,
        "daily_max": float(daily_max) if daily_max is not None else None,
        "daily_min": float(daily_min) if daily_min is not None else None,
        "daily_precipitation": float(daily_precip) if daily_precip is not None else None,
        "daily_wind_max": float(daily_wind) if daily_wind is not None else None,
        "source": "live",
    }

    return {
        "features": features,
        "report": weather_report,
    }


@app.route("/location-info")
def location_info():
    selected_state = request.args.get("state")
    selected_district = request.args.get("district")
    if not selected_state or not selected_district:
        return jsonify({"error": "State and district are required."}), 400

    percentages = get_location_percentages(selected_state, selected_district)
    history = get_history(selected_state, selected_district)

    features = get_location_features(selected_state, selected_district)
    weather_source = "average"
    weather_report = None
    live_weather = fetch_live_weather(selected_state, selected_district)
    if live_weather is not None:
        features = live_weather["features"]
        weather_report = live_weather["report"]
        weather_source = "live"
    elif features is not None:
        weather_source = "average"
    else:
        weather_source = "unknown"

    history_counts = get_previous_history_counts(selected_state, selected_district)
    predictions = predict_probabilities(features, percentages, history_counts)
    top_prediction = None
    if predictions:
        top_prediction = max(predictions.items(), key=lambda item: item[1])

    highest_historical = None
    if percentages:
        highest_historical = max(percentages.items(), key=lambda item: item[1])

    history_text = "No recent records found for this location."
    if history:
        latest = history[0]
        history_text = f"{latest['year']} {latest['disaster_type']} — {latest['count']} records"

    feature_text = "Weather data unavailable."
    if features:
        feature_text = (
            f"Temperature {features[0]}°C · Rainfall {features[1]}mm · "
            f"Humidity {features[2]}% · Wind {features[3]} km/h · Soil {features[4]}%"
        )

    return jsonify({
        "adaptive": f"Theme adapts to {selected_district}, {selected_state} using location context.",
        "history": {"summary": history_text},
        "history_summary": history_text,
        "smart": (
            f"Top predicted hazard is {top_prediction[0]} at {top_prediction[1]}%."
            if top_prediction
            else "Smart prediction is not available."
        ),
        "live_forecast": (
            f"{weather_source.capitalize()} forecast data: {feature_text}"
            if weather_source != "unknown"
            else "Forecast data is not available."
        ),
        "risk_mapping": (
            f"Highest historical risk: {highest_historical[0]} at {highest_historical[1]}%."
            if highest_historical
            else "Risk mapping data is not available."
        ),
        "ai_prediction": (
            f"AI predicts {top_prediction[0]} as the main hazard."
            if top_prediction
            else "AI prediction is not available."
        ),
        "predictions": predictions,
        "weather_report": weather_report,
        "weather_source": weather_source,
        "location": {
            "state": selected_state,
            "district": selected_district,
        },
        "report_meta": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "updated_from": weather_source,
        },
    })


@app.route("/", methods=["GET", "POST"])
def index():
    locations = get_locations()
    district_map = get_district_map()
    if request.method == "POST":
        selected_state = request.form.get("state")
        selected_district = request.form.get("district")
        if not selected_state or not selected_district:
            flash("Please select both a state and a district before predicting.")
            return redirect(url_for("index"))

        percentages = get_location_percentages(selected_state, selected_district)
        manual_features = parse_manual_features(request.form)
        weather_source = "manual"
        used_default = False
        used_live = False
        weather_report = None

        if manual_features is None:
            live_weather = fetch_live_weather(selected_state, selected_district)
            if live_weather is not None:
                manual_features = live_weather["features"]
                weather_source = "live"
                used_live = True
            else:
                manual_features = get_location_features(selected_state, selected_district)
                weather_source = "average"
                used_default = True

        history_counts = get_previous_history_counts(selected_state, selected_district)
        predictions = predict_probabilities(manual_features, percentages, history_counts)
        history = get_history(selected_state, selected_district)
        time_estimates = estimate_time_to_occurrence(predictions, percentages)
        feature_cards = build_feature_cards(
            selected_state,
            selected_district,
            predictions,
            percentages,
            history,
            weather_source,
            manual_features,
        )
        top_prediction = max(predictions.items(), key=lambda item: item[1])[0] if predictions else None
        safety_guidance = get_safety_guidance(top_prediction)
        return render_template(
            "result.html",
            state=selected_state,
            district=selected_district,
            percentages=percentages,
            history=history,
            predictions=predictions,
            time_estimates=time_estimates,
            used_default=used_default,
            used_live=used_live,
            weather_source=weather_source,
            features=manual_features,
            feature_cards=feature_cards,
            safety_guidance=safety_guidance,
            top_prediction=top_prediction,
            weather_report=weather_report,
        )

    return render_template("index.html", locations=locations, district_map=district_map)


if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "0.0.0.0")
    port = int(os.environ.get("APP_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    if debug:
        app.run(debug=True, host=host, port=port)
    else:
        print(f"Starting Disaster Predictor on {APP_URL} (host={host}, port={port})")
        try:
            from waitress import serve
            serve(app, host=host, port=port)
        except ImportError:
            app.run(debug=False, host=host, port=port)
