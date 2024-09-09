import streamlit as st
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Water Level Dashboard',
    page_icon=':droplet:',  # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_water_level_data():
    """Grab water level data from a CSV file.

    This uses caching to avoid having to read the file every time. If we were
    reading from an HTTP endpoint instead of a file, it's a good idea to set
    a maximum age to the cache with the TTL argument: @st.cache_data(ttl='1d')
    """
    DATA_FILENAME = Path(__file__).parent / 'src/data/data_predicted_PPB.csv'
    raw_data_df = pd.read_csv(DATA_FILENAME, parse_dates=['DATE_GMT'])
    
    return raw_data_df

water_level_df = get_water_level_data()

# Convert DATE_GMT to datetime.date
water_level_df['DATE_GMT'] = water_level_df['DATE_GMT'].dt.date

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.title('Water Level PoC')

# Add some spacing
st.write('')

# Add the clickable Google Maps link
st.markdown(
    """
    Data Location (Google Map): [link](https://www.google.com/maps/place/11%C2%B033'46.3%22N+104%C2%B056'07.5%22E/@11.5628639,104.9328323,17z/data=!3m1!4b1!4m4!3m3!8m2!3d11.5628587!4d104.9354126?entry=tts&g_ep=EgoyMDI0MDkwNC4wKgBIAVAD).
    """
)

# Filter the data by date range
min_date = water_level_df['DATE_GMT'].min()
max_date = water_level_df['DATE_GMT'].max()

from_date, to_date = st.slider(
    'Select the date range:',
    min_value=min_date,
    max_value=max_date,
    value=[min_date, max_date],
    format="YYYY-MM-DD"
)

# Filter data based on selected date range
filtered_data_df = water_level_df[
    (water_level_df['DATE_GMT'] >= from_date)
    & (water_level_df['DATE_GMT'] <= to_date)
]


# Classify the risk
def classify_risk(level):
    if level > 8:
        return 'High Risk'
    elif 5 < level <= 8:
        return 'Medium Risk'
    else:
        return 'Low Risk'

# Display the latest water levels and predictions
latest_data = filtered_data_df.iloc[-1]

# Determine the risk classification
risk_classification = classify_risk(latest_data["water_level"])

# Add a horizontal line
st.markdown("---")

# Create columns for the metric and risk classification
col1, col2 = st.columns([3, 2])  # Adjust the ratios based on your layout needs

with col1:
    today_date = datetime.today().strftime('%Y-%m-%d')
    st.metric(
        label=f'Latest Water Level ({today_date})',
        value=f'{latest_data["water_level"]:.2f}',
        delta=f'{latest_data["Predicted_Water_Level"]:.2f}',
        delta_color='normal'
    )

with col2:
    if risk_classification == 'High Risk':
        st.markdown(f"<h4 style='color: red;'>**Risk Classification:** {risk_classification}</h5>", unsafe_allow_html=True)
    elif risk_classification == 'Medium Risk':
        st.markdown(f"<h4 style='color: orange;'>**Risk Classification:** {risk_classification}</h4>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h4 style='color: green;'>**Risk Classification:** {risk_classification}</h4>", unsafe_allow_html=True)

# Add a horizontal line
st.markdown("---")


# Display water levels and predictions
st.line_chart(
    filtered_data_df.set_index('DATE_GMT')[['water_level', 'Predicted_Water_Level']],
    color=["#FF0000", "#0000FF"]
)


# Optional: Display historical and predicted water levels in a table
st.write("Historical and Predicted Water Levels")
st.dataframe(filtered_data_df[['DATE_GMT', 'water_level', 'Predicted_Water_Level']])


# Create a risk classification table
risk_table_df = pd.DataFrame({
    'Classification': ['High Risk', 'Medium Risk', 'Low Risk'],
    'Description': ['Water level > 8', '5 < Water level ≤ 8', 'Water level ≤ 5'],
    'Color': ['red', 'orange', 'green']
})

# Display the risk classification table
st.write("**Risk Classification Table (For Demo Purpose only)**")
st.dataframe(risk_table_df.style.applymap(lambda x: f'background-color: {x}' if x in ['red', 'orange', 'green'] else ''))
