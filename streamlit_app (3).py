import streamlit as st
import pandas as pd
import numpy as np
import requests
import altair as alt

st.set_page_config(page_title="PV & Netz Analyse", layout="wide")
st.title("🔋 PV-Erzeugung, Netzbezug & Batterie-Simulation")

# Sidebar: CSV Upload & Parameter Input
st.sidebar.header("📁 CSV Upload & Einstellungen")
nutzungsbereich = st.sidebar.selectbox(
    "Analyse-Bereich wählen", ["Tagesanalyse", "Wochenanalyse", "Monatsanalyse"]
)
start_date = st.sidebar.date_input("Startdatum", value=pd.to_datetime('2024-01-01'))
end_date = st.sidebar.date_input("Enddatum", value=pd.to_datetime('2024-01-31'))

netz_csv = st.sidebar.file_uploader("1️⃣ CSV Netzbezug (datetime, power_kw)", type="csv")
pv_csvs = st.sidebar.file_uploader(
    "2️⃣ CSV PV-Erzeugung (datetime, power_kw)",
    type="csv",
    accept_multiple_files=True
)

# PVGIS Parameter
st.sidebar.subheader("🌞 PVGIS-PARAMETER für Performance Ratio")
lat = st.sidebar.number_input("Latitude", value=48.2082)
lon = st.sidebar.number_input("Longitude", value=16.3738)
year = st.sidebar.number_input("Jahr für PVGIS-Daten", min_value=2000, max_value=2025, value=2024)

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
        theoretical = yearly_irr  # kWh/kWp Jahresertrag
        pr = actual_energy / theoretical * 100 if theoretical > 0 else np.nan
        return pr
    except Exception:
        return np.nan

# Batterie-Simulation
def simulate_battery(df, battery_capacity, effizienz):
    """Simuliert Ladevorgang und Entladung eines Batteriespeichers"""
    soc = 0.0
    soc_list = []
    for _, row in df.iterrows():
        surplus = row['PV_gesamt_kWh'] - row['Eigenverbrauch_kWh']
        demand = row['Netzbezug_kWh']
        # Laden
        if surplus > 0:
            charge = min(surplus, battery_capacity - soc) * effizienz
            soc += charge
        # Entladen
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
        st.info("Bitte CSVs hochladen, um die Analyse zu starten.")
        return

    # Daten einlesen & umrechnen
    df_netz = pd.read_csv(netz_csv)
    df_netz = leistung_zu_energie_fest(df_netz)
    df_netz = df_netz.rename(columns={'energie_kWh': 'Netzbezug_kWh'})

    pv_frames = []
    for f in pv_csvs:
        df = pd.read_csv(f)
        df = leistung_zu_energie_fest(df)
        df = df.rename(columns={'energie_kWh': f.name})
        pv_frames.append(df)

    # PV kombinieren
    df_pv = pd.concat(pv_frames).groupby('datetime').sum().reset_index()
    df_pv['PV_gesamt_kWh'] = df_pv.drop(columns=['datetime']).sum(axis=1)

    # Merge Netz & PV
    df_all = pd.merge(
        df_pv[['datetime','PV_gesamt_kWh']],
        df_netz,
        on='datetime',
        how='outer'
    ).fillna(0)

    # Filter Zeitbereich
    df_all['datetime'] = pd.to_datetime(df_all['datetime'])
    mask = (
        (df_all['datetime'].dt.date >= start_date) &
        (df_all['datetime'].dt.date <= end_date)
    )
    df_all = df_all.loc[mask].sort_values('datetime')

    # Datenqualität: fehlende Timestamps identifizieren
    full_range = pd.date_range(
        start=df_all['datetime'].min(),
        end=df_all['datetime'].max(),
        freq='15T'
    )
    missing = full_range.difference(df_all['datetime'])
    if len(missing) > 0:
        st.warning(f"Es fehlen {len(missing)} Zeitstempel in den Daten.")

    # Anomalie-Erkennung (IQR Outlier)
    q1, q3 = df_all['PV_gesamt_kWh'].quantile([0.25,0.75])
    iqr = q3 - q1
    outliers = df_all[
        (df_all['PV_gesamt_kWh'] < q1 - 1.5*iqr) |
        (df_all['PV_gesamt_kWh'] > q3 + 1.5*iqr)
    ]
    if not outliers.empty:
        st.warning(f"{len(outliers)} Ausreißer in PV-Erzeugung erkannt.")

    # KPIs: Eigenverbrauch, Einspeisung, Autarkie
    df_all['Eigenverbrauch_kWh'] = np.minimum(
        df_all['PV_gesamt_kWh'], df_all['Netzbezug_kWh']
    )
    df_all['Einspeisung_kWh'] = (
        df_all['PV_gesamt_kWh'] - df_all['Eigenverbrauch_kWh']
    )

    total = df_all[
        ['PV_gesamt_kWh','Netzbezug_kWh','Eigenverbrauch_kWh','Einspeisung_kWh']
    ].sum()
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

    # 1. Zeitreihen-Aggregation
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

    # 2. Ausreißer & Verteilung
    st.subheader("🚨 Ausreißer & Verteilung")
    st.dataframe(outliers)
    hist = alt.Chart(df_all.reset_index()).mark_bar().encode(
        alt.X('PV_gesamt_kWh', bin=alt.Bin(maxbins=50)),
        y='count()'
    )
    st.altair_chart(hist, use_container_width=True)

    # 3. Stacked Area Chart: PV vs Eigenverbrauch vs Einspeisung
    st.subheader("📊 Stacked Area Chart")
    area_df = df_all.reset_index()[
        ['datetime','PV_gesamt_kWh','Eigenverbrauch_kWh','Einspeisung_kWh']
    ]
    area = alt.Chart(area_df).transform_fold(
        ['PV_gesamt_kWh','Eigenverbrauch_kWh','Einspeisung_kWh'],
        as_=['Kategorie','kWh']
    ).mark_area(opacity=0.5).encode(
        x='datetime:T',
        y='kWh:Q',
        color='Kategorie:N'
    )
    st.altair_chart(area, use_container_width=True)

    # Batterie-Simulation
    simulate_battery(df_all, battery_capacity, effizienz)


if __name__ == "__main__":
    main()
