import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np

# 1. KONFIGURACJA STRONY (Musi być na samej górze)
st.set_page_config(page_title="InPost City Analyzer", page_icon="📦", layout="wide")

# 2. ŁADOWANIE DANYCH (Z cache'owaniem, żeby aplikacja nie ładowała pliku przy każdym kliknięciu)
@st.cache_data
def load_data():
    # Podmień na nazwę swojego ostatecznego pliku CSV
    df = pd.read_csv("inpost_parcel_locker.csv")
    
    # Upewnijmy się, że 'is_doubled' jest intem, a jeśli z jakiegoś powodu masz nadal 'apm_doubled', to zróbmy flagę tu:
    if 'apm_doubled' in df.columns:
        df['is_doubled'] = df['apm_doubled'].notna().astype(int)
    
    # Funkcja do mapowania poziomu smogu na kolory RGB [R, G, B, Opacity]
    def get_color(air_index):
        if air_index ==  "VERY_GOOD":
            return [46, 204, 113, 200]  # Zielony (Flota spalinowa OK)
        elif air_index ==  "GOOD":
            return [241, 196, 15, 200]  # Żółty/Pomarańczowy (Ostrzeżenie)
        elif air_index ==  "VERY_BAD":
            return [231, 76, 60, 200]   # Czerwony (Wymagane EV!)
        else:
            return [149, 165, 166, 200] # Szary (Brak danych)
            
    df['color'] = df['air_index_level'].apply(get_color)
    
    # Rozmiar kropki na mapie na podstawie gabarytu maszyny (typ 3.0, 4.0, 6.0)
    # Mnożymy x15, żeby promień kropki w metrach był widoczny na mapie
    df['radius'] = df['physical_type_mapped'].fillna(4.0) * 15 
    
    # Jeśli nie ma na liście zamienników, dajemy ładny komunikat
    df['recommended_low_interest_box_machines_list'] = df['recommended_low_interest_box_machines_list'].fillna("Brak zaleceń - brak przeciążenia")
    
    return df

df = load_data()

# 3. INTERFEJS UŻYTKOWNIKA - PANEL BOCZNY (Sidebar)
st.sidebar.image("https://inpost.pl/sites/default/files/inpost-logo.png", width=150)
st.sidebar.title("🎛️ Panel Dyspozytora")
st.sidebar.markdown("Zarządzaj wskaźnikami Green-SLA oraz przepustowością sieci w czasie rzeczywistym.")

# Filtry
st.sidebar.header("Filtry")

# Filtr: Przeciążenie
show_congested_only = st.sidebar.checkbox("⚠️ Pokaż tylko przeciążone (is_doubled == 1)", value=False)

# Filtr: Smog
air_filter = st.sidebar.multiselect(
    "🌬️ Filtruj Jakość Powietrza:",
    options=df['air_index_level'].dropna().unique(),
    default=df['air_index_level'].dropna().unique()
)

# Filtr: Indoor / Outdoor
location_type_filter = st.sidebar.radio(
    "🏢 Typ Lokalizacji (Okna Czasowe):",
    options=["Wszystkie", "Outdoor (Elastyczne)", "Indoor (Twarde)"]
)

# 4. LOGIKA FILTROWANIA DANYCH
filtered_df = df.copy()

if show_congested_only:
    filtered_df = filtered_df[filtered_df['is_doubled'] == 1]
    
if air_filter:
    filtered_df = filtered_df[filtered_df['air_index_level'].isin(air_filter)]

if location_type_filter == "Outdoor (Elastyczne)":
    filtered_df = filtered_df[filtered_df['location_type'] == 'Outdoor']
elif location_type_filter == "Indoor (Twarde)":
    filtered_df = filtered_df[filtered_df['location_type'] == 'Indoor']

# 5. WIDOK GŁÓWNY - DASHBOARD
st.title("🗺️ InPost City Analyzer - Kraków")
st.markdown("Mapa przepustowości sieci Paczkomatów i optymalizacji floty pod kątem emisji CO2.")

# KPI (Kluczowe Wskaźniki Efektywności) na górze
col1, col2, col3, col4 = st.columns(4)
col1.metric("Wszystkie widoczne maszyny", len(filtered_df))
col2.metric("Maszyny z wysokim Smogiem", len(filtered_df[filtered_df['air_index_level'].isin(['POOR', 'VERY_POOR'])]))
col3.metric("Maszyny Indoor (Trudne SLA)", len(filtered_df[filtered_df['location_type'] == 'Indoor']))
col4.metric("Punkty krytyczne (Zatory)", len(filtered_df[filtered_df['is_doubled'] == 1]))

# 6. RENDEROWANIE MAPY W PYDECK
# Ustawiamy środek mapy na podstawie średnich współrzędnych z danych
midpoint = (np.average(filtered_df['location_latitude']), np.average(filtered_df['location_longitude'])) if not filtered_df.empty else (50.06, 19.94)

# Definiujemy warstwę punktową (Scatterplot)
layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_df,
    get_position='[location_longitude, location_latitude]',
    get_color='color',
    get_radius='radius',
    pickable=True,
    opacity=0.8,
    stroked=True,
    filled=True,
    radius_scale=1,
    radius_min_pixels=3,
    radius_max_pixels=30,
    line_width_min_pixels=1,
)

# Konfigurujemy widok mapy
view_state = pdk.ViewState(
    latitude=midpoint[0],
    longitude=midpoint[1],
    zoom=11.5,
    pitch=45, # Pochylenie mapy dla fajnego efektu 3D
)

# Wyświetlamy mapę z Tooltipem
st.pydeck_chart(pdk.Deck(
    map_provider='carto',            # Używamy darmowego dostawcy map
    map_style='dark',                # Styl mapy (dark lub light)
    initial_view_state=view_state,
    layers=[layer],
    tooltip={
        "html": "<b>ID:</b> {name} <br/>"
                "<b>Smog:</b> {air_index_level} <br/>"
                "<b>Typ:</b> {location_type} (Rozmiar: {physical_type_mapped}) <br/>"
                "<b>Przeciążona?:</b> {is_doubled} <br/>"
                "<hr/>"
                "<b>➡️ Rekomendowane przekierowanie:</b> <br/> {recommended_low_interest_box_machines_list}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "font-family": "Helvetica, Arial, sans-serif"
        }
    }
))

# 7. TABELA Z DANYMI (Opcjonalnie pod mapą)
with st.expander("Kliknij, aby rozwinąć surowe dane analityczne"):
    st.dataframe(filtered_df[['name', 'location_type', 'physical_type_mapped', 'is_doubled', 'air_index_level', 'recommended_low_interest_box_machines_list']])