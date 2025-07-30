import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
import datetime as dt

st.set_page_config(page_title="PV & Netz Analyse", layout="wide")
st.title("🔋 PV-Erzeugung, Netzbezug & Batterie-Simulation")

# Sidebar: CSV Upload & Einstellungen
st.sidebar.header("📁 CSV Upload & Einstellungen")
nutzungsbereich = st.sidebar.selectbox(
    "Analyse-Bereich wählen", ["Tagesanalyse", "Wochenanalyse", "Monatsanalyse"]
)

# PVGIS & Zeitraum
st.sidebar.subheader("🌞 Zeitraum & PVGIS")
year = st.sidebar.number_input(
    "Jahr für Analyse & PVGIS-Daten", min_value=2000, max_value=2025, value=2024
)
full_year = st.sidebar.checkbox("Gesamtes Jahr analysieren", value=False)
if full_year:
    start_date = pd.Timestamp(year=year, month=1, day=1)
    end_date = pd.Timestamp(year=year, month=12, day=31)
else:
    start_date = st.sidebar.date_input("Startdatum", value=pd.Timestamp(year,1,1))
    end_date = st.sidebar.date_input("Enddatum", value=pd.Timestamp(year,1,31))

# Ort oder manuelle Koordinaten
st.sidebar.subheader("📍 Standort für PVGIS")
location_input = st.sidebar.text_input("Ort (z.B. Wien, Austria)", "Wien, Austria")
lat = None
lon = None
if location_input:
    geolocator = Nominatim(user_agent="pvgis_app")
    try:
        location = geolocator.geocode(location_input, timeout=10)
    except GeocoderUnavailable:
        location = None
    if location:
        lat, lon = location.latitude, location.longitude
    else:
        st.sidebar.warning("Geocoding fehlgeschlagen. Bitte Koordinaten manuell eingeben.")
        lat = st.sidebar.number_input("Latitude", value=48.2082, format="%.6f")
        lon = st.sidebar.number_input("Longitude", value=16.3738, format="%.6f")
else:
    st.sidebar.warning("Bitte Ort oder Koordinaten angeben.")
    lat = st.sidebar.number_input("Latitude", value=48.2082, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=16.3738, format="%.6f")

netz_csv = st.sidebar.file_uploader("1️⃣ CSV Netzbezug (datetime, power_kw)", type="csv")
pv_csvs = st.sidebar.file_uploader(
    "2️⃣ CSV PV-Erzeugung (datetime, power_kw)",
    type="csv",
    accept_multiple_files=True
)

# Batterie Parameter
st.sidebar.subheader("🔋 Batteriespeicher Einstellungen")
battery_capacity = st.sidebar.number_input("Kapazität (kWh)", min_value=0.0, value=10.0)
effizienz = st.sidebar.slider("Wirkungsgrad (Roundtrip)", 0.0, 1.0, 0.9)

# Hilfsfunktion: kW -> kWh (15min Intervall)
INTERVALL_H = 0.25
def leistung_zu_energie_fest(df, leistung_col='power_kw', zeit_col='datetime'):
    df[zeit_col] = pd.to_datetime(df[zeit_col])
    df['energie_kWh'] = df[leistung_col] * INTERVALL_H
    return df[[zeit_col, 'energie_kWh']]

# PVGIS Performance Ratio
def calculate_performance_ratio(actual_energy, lat, lon, year):
    """Ruft PVGIS API auf und berechnet Performance Ratio"""
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/Tmy?lat={lat}&lon={lon}&outputformat=json"
    try:
        resp = requests.get(url)
        data = resp.json()
        yearly_irr = sum([m['H(i)'] for m in data['outputs']['monthly']])
        theoretical = yearly_irr
        return actual_energy / theoretical * 100 if theoretical > 0 else np.nan
    except Exception:
        return np.nan

# Batterie-Simulation
def simulate_battery(df, battery_capacity, effizienz):
    soc = 0.0
    soc_list = []
    for _, row in df.iterrows():
        surplus = row['PV_gesamt_kWh'] - row['Eigenverbrauch_kWh']
        demand = row['Netzbezug_kWh']
        if surplus > 0:
            charge = min(surplus, battery_capacity - soc) * effizienz
            soc += charge
        if demand > 0 and soc > 0:
            drawn = min(demand, soc) * effizienz
            soc -= drawn
        soc_list.append(soc)
    df['SoC_kWh'] = soc_list
    st.subheader("🔋 Batterie State of Charge (SoC)")
    st.line_chart(df['SoC_kWh'])

# Haupt-Workflow
def main():
    if not (netz_csv and pv_csvs):
        st.info("Bitte CSVs hochladen, um Analyse zu starten.")
        return

    # Daten einlesen & umrechnen
    df_netz = pd.read_csv(netz_csv)
    df_netz = leistung_zu_energie_fest(df_netz)
    df_netz.rename(columns={'energie_kWh':'Netzbezug_kWh'}, inplace=True)

    pv_frames = []
    for f in pv_csvs:
        df = pd.read_csv(f)
        df = leistung_zu_energie_fest(df)
        df.rename(columns={'energie_kWh':f.name}, inplace=True)
        pv_frames.append(df)

    df_pv = pd.concat(pv_frames).groupby('datetime').sum().reset_index()
    df_pv['PV_gesamt_kWh'] = df_pv.drop(columns=['datetime']).sum(axis=1)

    df_all = pd.merge(
        df_pv[['datetime','PV_gesamt_kWh']], df_netz,
        on='datetime', how='outer'
    ).fillna(0)

    df_all['datetime'] = pd.to_datetime(df_all['datetime'])
    mask = (
        (df_all['datetime'].dt.date >= pd.to_datetime(start_date).date()) &
        (df_all['datetime'].dt.date <= pd.to_datetime(end_date).date())
    )
    df_all = df_all.loc[mask].sort_values('datetime')

    # Datenqualität & Anomalien
    full_range = pd.date_range(start=df_all['datetime'].min(),
                               end=df_all['datetime'].max(),
                               freq='15T')
    missing = full_range.difference(df_all['datetime'])
    if missing.any():
        st.warning(f"Es fehlen {len(missing)} Zeitstempel.")
    q1, q3 = df_all['PV_gesamt_kWh'].quantile([0.25,0.75])
    outliers = df_all[
        (df_all['PV_gesamt_kWh'] < q1 - 1.5*(q3-q1)) |
        (df_all['PV_gesamt_kWh'] > q3 + 1.5*(q3-q1))
    ]
    if not outliers.empty:
        st.warning(f"{len(outliers)} Ausreißer erkannt.")

    # KPIs
    df_all['Eigenverbrauch_kWh'] = np.minimum(
        df_all['PV_gesamt_kWh'], df_all['Netzbezug_kWh']
    )
    df_all['Einspeisung_kWh'] = (
        df_all['PV_gesamt_kWh'] - df_all['Eigenverbrauch_kWh']
    )
    total = df_all[['PV_gesamt_kWh','Netzbezug_kWh',
                    'Eigenverbrauch_kWh','Einspeisung_kWh']].sum()
    pr_ratio = calculate_performance_ratio(
        total['PV_gesamt_kWh'], lat, lon, year
    )

    st.subheader("🔎 Kennzahlen Gesamt")
    cols = st.columns(5)
    cols[0].metric("PV Erzeugung (kWh)", f"{total['PV_gesamt_kWh']:.1f}")
    cols[1].metric("Netzbezug (kWh)", f"{total['Netzbezug_kWh']:.1f}")
    cols[2].metric("Eigenverbrauch (kWh)", f"{total['Eigenverbrauch_kWh']:.1f}")
    cols[3].metric("Einspeisung (kWh)", f"{total['Einspeisung_kWh']:.1f}")
    cols[4].metric("Performance Ratio (PR)", f"{pr_ratio:.1f} %")

    # Aggregationen
    df_all.set_index('datetime', inplace=True)
    if nutzungsbereich == 'Tagesanalyse':
        df_aggr = df_all.resample('D').sum()
    elif nutzungsbereich == 'Wochenanalyse':
        df_aggr = df_all.resample('W-MON').sum()
    else:
        df_aggr = df_all.resample('M').sum()
    df_aggr['Autarkie_%'] = (
        df_aggr['Eigenverbrauch_kWh'] /
        df_aggr['Netzbezug_kWh'] * 100
    )
    st.subheader(f"📈 {nutzungsbereich}")
    st.line_chart(df_aggr[['PV_gesamt_kWh','Netzbezug_kWh']])
    st.bar_chart(df_aggr['Autarkie_%'])

    # Visualisierungen
    st.subheader("🚨 Ausreißer & Verteilung")
    st.dataframe(outliers)
    hist = alt.Chart(df_all.reset_index()).mark_bar().encode(
        alt.X('PV_gesamt_kWh', bin=alt.Bin(maxbins=50)),
        y='count()'
    )
    st.altair_chart(hist, use_container_width=True)

    st.subheader("📊 Stacked Area Chart")
    area_df = df_all.reset_index()[['datetime','PV_gesamt_kWh',
                                     'Eigenverbrauch_kWh',
                                     'Einspeisung_kWh']]
    area = alt.Chart(area_df).transform_fold(
        ['PV_gesamt_kWh','Eigenverbrauch_kWh','Einspeisung_kWh'],
        as_=['Kategorie','kWh']
    ).mark_area(opacity=0.5).encode(
        x='datetime:T',
        y='kWh:Q',
        color='Kategorie:N'
    )
    st.altair_chart(area, use_container_width=True)

    simulate_battery(df_all, battery_capacity, effizienz)

if __name__ == "__main__":
    main()
