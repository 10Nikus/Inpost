import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def generate_resource_constrained_zones(input_csv: str, output_csv: str, n_zones: int = 5):
    """
    K-Means połączony z algorytmem zachłannym do alokacji limitowanej floty.
    """
    df = pd.read_csv(input_csv)
    
    # 1. K-MEANS KLASTERZYZACJA
    coords = df[['location_latitude', 'location_longitude']]
    kmeans = KMeans(n_clusters=n_zones, random_state=42, n_init=10)
    df['delivery_zone_id'] = kmeans.fit_predict(coords)

    # Mapowanie smogu
    smog_mapping = {'VERY_GOOD': 1, 'GOOD': 2, 'VERY_BAD': 3}
    df['air_score'] = df['air_index_level'].map(smog_mapping)

    # 2. STATYSTYKI STREF (Zapotrzebowanie)
    zone_stats = df.groupby('delivery_zone_id').agg(
        avg_smog=('air_score', 'mean'),
        machine_count=('name', 'count')
    ).reset_index()

    # --- NOWA LOGIKA BIZNESOWA: ALOKACJA ZASOBÓW ---
    
    X_TOTAL_CARS = 50
    Y_EV = 25
    Z_HYBRID = 10
    ICE_DIESEL = X_TOTAL_CARS - Y_EV - Z_HYBRID
    
    # Wyliczamy ile aut (rejonów) potrzebuje każda strefa proporcjonalnie do liczby maszyn
    total_machines = zone_stats['machine_count'].sum()
    zone_stats['cars_needed'] = np.ceil((zone_stats['machine_count'] / total_machines) * X_TOTAL_CARS).astype(int)
    
    # Sortujemy strefy od NAJGORSZEGO smogu (descending), żeby tam najpierw wysłać elektryki
    zone_stats = zone_stats.sort_values(by='avg_smog', ascending=False).reset_index(drop=True)
    
    allocations = []
    available_ev = Y_EV
    available_hybrid = Z_HYBRID
    available_ice = ICE_DIESEL

    # Algorytm Zachłanny (Rozdawanie aut)
    for idx, row in zone_stats.iterrows():
        needed = row['cars_needed']
        ev_alloc = 0
        hybrid_alloc = 0
        ice_alloc = 0
        
        # Najpierw dajemy elektryki, jeśli je mamy
        if available_ev > 0:
            take = min(needed, available_ev)
            ev_alloc = take
            available_ev -= take
            needed -= take
            
        # Potem hybrydy
        if needed > 0 and available_hybrid > 0:
            take = min(needed, available_hybrid)
            hybrid_alloc = take
            available_hybrid -= take
            needed -= take
            
        # Na końcu dobijamy dieslami
        if needed > 0:
            take = min(needed, available_ice) # teoretycznie może nam braknąć diesli przez zaokrąglenia, ale upraszczamy
            ice_alloc = take
            available_ice -= take
            needed -= take

        # Definiujemy "główny profil floty" dla danej strefy na podstawie tego, czego dostała najwięcej
        if ev_alloc >= hybrid_alloc and ev_alloc >= ice_alloc:
            profile = "Dominacja EV"
        elif hybrid_alloc > ev_alloc and hybrid_alloc >= ice_alloc:
            profile = "Strefa Mieszana (Hybrydy)"
        else:
            profile = "Strefa Spalinowa"

        allocations.append({
            'delivery_zone_id': row['delivery_zone_id'],
            'avg_smog': row['avg_smog'],
            'machine_count': row['machine_count'],
            'cars_needed': row['cars_needed'],
            'EV_assigned': ev_alloc,
            'Hybrid_assigned': hybrid_alloc,
            'Diesel_assigned': ice_alloc,
            'fleet_profile': profile
        })

    alloc_df = pd.DataFrame(allocations)
    
    print(f"\n📦 --- WYNIKI OPTYMALIZACJI FLOTY DLA KRAKOWA (X={X_TOTAL_CARS}, Y={Y_EV}, Z={Z_HYBRID}) ---")
    print(alloc_df[['delivery_zone_id', 'avg_smog', 'cars_needed', 'EV_assigned', 'Hybrid_assigned', 'Diesel_assigned', 'fleet_profile']])
    
    # 3. ŁĄCZENIE I ZAPIS
    # Dodajemy przypisany profil do oryginalnych danych o maszynach
    df = df.merge(alloc_df[['delivery_zone_id', 'fleet_profile']], on='delivery_zone_id', how='left')
    df.drop(columns=['air_score'], inplace=True, errors='ignore')
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    generate_resource_constrained_zones("inpost_parcel_locker.csv", "inpost_with_zones.csv")