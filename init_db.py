import os
import sqlite3
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(__file__)
CSV_PATH = os.path.join(BASE_DIR, "disaster_dataset.csv")
DB_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DB_DIR, "disaster.db")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

STATES = [
    "Andaman & Nicobar",
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chandigarh",
    "Chhattisgarh",
    "Dadra & Nagar Haveli & Daman & Diu",
    "Delhi",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jammu & Kashmir",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Ladakh",
    "Lakshadweep",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Puducherry",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
]

DISTRICT_MAP = {
    "Andaman & Nicobar": ["Port Blair"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Tirupati", "Guntur", "Nellore"],
    "Arunachal Pradesh": ["Itanagar", "Tawang", "Tezu"],
    "Assam": ["Guwahati", "Dibrugarh", "Silchar", "Jorhat", "Tezpur"],
    "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur", "Darbhanga"],
    "Chandigarh": ["Chandigarh"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bhilai", "Jagdalpur"],
    "Dadra & Nagar Haveli & Daman & Diu": ["Silvassa", "Daman"],
    "Delhi": ["New Delhi"],
    "Goa": ["Panaji", "Margao", "Vasco da Gama"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Haryana": ["Chandigarh", "Gurugram", "Panipat", "Faridabad", "Ambala"],
    "Himachal Pradesh": ["Shimla", "Dharamshala", "Manali", "Kullu", "Solan"],
    "Jammu & Kashmir": ["Srinagar", "Jammu", "Anantnag"],
    "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Dumka", "Bokaro"],
    "Karnataka": ["Bengaluru", "Mysore", "Mangalore", "Hubli", "Belgaum"],
    "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur", "Kannur"],
    "Ladakh": ["Leh", "Kargil"],
    "Lakshadweep": ["Kavaratti"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad"],
    "Manipur": ["Imphal", "Thoubal", "Ukhrul"],
    "Meghalaya": ["Shillong", "Tura", "Jowai"],
    "Mizoram": ["Aizawl", "Lunglei", "Champhai"],
    "Nagaland": ["Kohima", "Dimapur", "Mokokchung"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Puri", "Berhampur"],
    "Puducherry": ["Puducherry", "Karaikal"],
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

FEATURE_COLUMNS = [
    "temperature",
    "rainfall",
    "humidity",
    "wind_speed",
    "soil_moisture",
]


def build_database():
    os.makedirs(DB_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Missing dataset file: {CSV_PATH}. Please copy disaster_dataset.csv into the project folder."
        )

    df = pd.read_csv(CSV_PATH)
    df = df.copy()
    df["year"] = 2018 + (df.index % 6)
    df["state"] = [STATES[i % len(STATES)] for i in range(len(df))]
    df["district"] = df["state"].map(lambda state: DISTRICT_MAP[state][0])

    for state in STATES:
        district_list = DISTRICT_MAP[state]
        state_mask = df["state"] == state
        indices = df[state_mask].index
        for i, idx in enumerate(indices):
            df.at[idx, "district"] = district_list[i % len(district_list)]

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS records")
        cursor.execute(
            """
            CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                temperature REAL,
                rainfall REAL,
                humidity REAL,
                wind_speed REAL,
                soil_moisture REAL,
                disaster_type TEXT,
                state TEXT,
                district TEXT,
                year INTEGER
            )
            """
        )

        insert_sql = (
            "INSERT INTO records (temperature, rainfall, humidity, wind_speed, soil_moisture, disaster_type, state, district, year) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )

        for row in df.itertuples(index=False):
            cursor.execute(
                insert_sql,
                (
                    float(row.temperature),
                    float(row.rainfall),
                    float(row.humidity),
                    float(row.wind_speed),
                    float(row.soil_moisture),
                    str(row.disaster_type),
                    str(row.state),
                    str(row.district),
                    int(row.year),
                ),
            )

        cursor.execute("DROP TABLE IF EXISTS state_stats")
        cursor.execute(
            """
            CREATE TABLE state_stats (
                state TEXT,
                disaster_type TEXT,
                record_count INTEGER,
                percentage REAL,
                PRIMARY KEY (state, disaster_type)
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS district_stats")
        cursor.execute(
            """
            CREATE TABLE district_stats (
                state TEXT,
                district TEXT,
                disaster_type TEXT,
                record_count INTEGER,
                percentage REAL,
                PRIMARY KEY (state, district, disaster_type)
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS state_features")
        cursor.execute(
            """
            CREATE TABLE state_features (
                state TEXT PRIMARY KEY,
                temperature REAL,
                rainfall REAL,
                humidity REAL,
                wind_speed REAL,
                soil_moisture REAL
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS district_features")
        cursor.execute(
            """
            CREATE TABLE district_features (
                state TEXT,
                district TEXT,
                temperature REAL,
                rainfall REAL,
                humidity REAL,
                wind_speed REAL,
                soil_moisture REAL,
                PRIMARY KEY (state, district)
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS history_summary")
        cursor.execute(
            """
            CREATE TABLE history_summary (
                state TEXT,
                district TEXT,
                year INTEGER,
                disaster_type TEXT,
                record_count INTEGER,
                PRIMARY KEY (state, district, year, disaster_type)
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS state_history")
        cursor.execute(
            """
            CREATE TABLE state_history (
                state TEXT,
                year INTEGER,
                disaster_type TEXT,
                record_count INTEGER,
                PRIMARY KEY (state, year, disaster_type)
            )
            """
        )

        cursor.execute("DROP TABLE IF EXISTS global_stats")
        cursor.execute(
            """
            CREATE TABLE global_stats (
                disaster_type TEXT PRIMARY KEY,
                percentage REAL
            )
            """
        )

        total_records = len(df)
        global_counts = df["disaster_type"].value_counts(normalize=True) * 100
        for disaster_type, percentage in global_counts.items():
            cursor.execute(
                "INSERT INTO global_stats (disaster_type, percentage) VALUES (?, ?)",
                (str(disaster_type), float(percentage)),
            )

        for state in df["state"].sort_values().unique():
            state_df = df[df["state"] == state]
            counts = state_df["disaster_type"].value_counts()
            total_by_state = len(state_df)
            for disaster_type, count in counts.items():
                percentage = count / total_by_state * 100
                cursor.execute(
                    "INSERT INTO state_stats (state, disaster_type, record_count, percentage) VALUES (?, ?, ?, ?)",
                    (state, str(disaster_type), int(count), float(percentage)),
                )

            feature_means = state_df[FEATURE_COLUMNS].mean().to_dict()
            cursor.execute(
                "INSERT INTO state_features (state, temperature, rainfall, humidity, wind_speed, soil_moisture) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    state,
                    float(feature_means["temperature"]),
                    float(feature_means["rainfall"]),
                    float(feature_means["humidity"]),
                    float(feature_means["wind_speed"]),
                    float(feature_means["soil_moisture"]),
                ),
            )

        for state in df["state"].sort_values().unique():
            state_df = df[df["state"] == state]
            for district in DISTRICT_MAP[state]:
                district_df = state_df[state_df["district"] == district]
                counts = district_df["disaster_type"].value_counts()
                total_by_district = len(district_df)
                for disaster_type, count in counts.items():
                    percentage = count / total_by_district * 100
                    cursor.execute(
                        "INSERT INTO district_stats (state, district, disaster_type, record_count, percentage) VALUES (?, ?, ?, ?, ?)",
                        (state, district, str(disaster_type), int(count), float(percentage)),
                    )

                feature_means = district_df[FEATURE_COLUMNS].mean().to_dict()
                cursor.execute(
                    "INSERT INTO district_features (state, district, temperature, rainfall, humidity, wind_speed, soil_moisture) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        state,
                        district,
                        float(feature_means["temperature"]),
                        float(feature_means["rainfall"]),
                        float(feature_means["humidity"]),
                        float(feature_means["wind_speed"]),
                        float(feature_means["soil_moisture"]),
                    ),
                )

        history_summary = df.groupby(["state", "district", "year", "disaster_type"]).size().reset_index(name="record_count")
        for row in history_summary.itertuples(index=False):
            cursor.execute(
                "INSERT INTO history_summary (state, district, year, disaster_type, record_count) VALUES (?, ?, ?, ?, ?)",
                (row.state, row.district, int(row.year), str(row.disaster_type), int(row.record_count)),
            )

        state_history = df.groupby(["state", "year", "disaster_type"]).size().reset_index(name="record_count")
        for row in state_history.itertuples(index=False):
            cursor.execute(
                "INSERT INTO state_history (state, year, disaster_type, record_count) VALUES (?, ?, ?, ?)",
                (row.state, int(row.year), str(row.disaster_type), int(row.record_count)),
            )

        conn.commit()

    print(f"Database created at: {DB_PATH}")
    print(f"States loaded: {', '.join(sorted(df['state'].unique()))}")
    print(f"Total records inserted: {total_records}")
    print(f"Districts loaded: {', '.join(sorted(set(df['district'].tolist())))}")

    train_model(df)


def train_model(df):
    X = df[FEATURE_COLUMNS].astype(float).values
    y = df["disaster_type"].astype(str)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y_encoded)

    training_accuracy = model.score(X, y_encoded)
    print(f"Model training completed. Training accuracy: {training_accuracy:.3f}")

    model_data = {
        "model": model,
        "classes": label_encoder.classes_.tolist(),
    }
    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model_data, model_file)

    print(f"Predictive model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    build_database()
