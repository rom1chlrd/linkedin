import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURATION ---
TICKER = "SPY"
# Exemples de dates de publication du CPI US (inflation) récentes
EVENT_DATES = [
    '2023-09-13', '2023-10-12', '2023-11-14', '2023-12-12',
    '2024-01-11', '2024-02-13', '2024-03-12', '2024-04-10'
]
WINDOW_BEFORE = 5  # Jours de trading avant l'événement
WINDOW_AFTER = 10   # Jours de trading après l'événement

# Mise en place du style Seaborn pour un rendu "pro" immédiat
sns.set_theme(style="whitegrid", context="talk")

# --- 1. RECUPERATION DES DONNEES ---
print(f"Téléchargement des données pour {TICKER}...")
# On prend une marge large pour être sûr d'avoir les fenêtres
start_date = pd.to_datetime(min(EVENT_DATES)) - pd.Timedelta(days=40)
end_date = pd.to_datetime(max(EVENT_DATES)) + pd.Timedelta(days=40)
data = yf.download(TICKER, start=start_date, end=end_date, progress=False)
data = data.reset_index() # S'assurer que 'Date' est une colonne

# --- 2. TRAITEMENT DES FENETRES D'EVENEMENTS ---
processed_events = []

print("Traitement des événements...")
for event_date_str in EVENT_DATES:
    event_date_dt = pd.to_datetime(event_date_str)
    
    # Trouver l'index le plus proche de la date de l'événement
    # (Gère les cas où l'événement tombe un week-end/férié)
    try:
        idx = data[data['Date'] >= event_date_dt].index[0]
    except IndexError:
        print(f"Données insuffisantes pour la date : {event_date_str}")
        continue

    # Définir les bornes de la fenêtre
    start_idx = max(0, idx - WINDOW_BEFORE)
    end_idx = min(len(data), idx + WINDOW_AFTER + 1)
    
    # Extraire la fenêtre
    window_df = data.iloc[start_idx:end_idx].copy()
    
    if len(window_df) < (WINDOW_BEFORE + WINDOW_AFTER):
        continue # Skip si fenêtre incomplète

    # --- NORMALISATION ---
    # Créer un axe X relatif : de -5 à +10
    window_df['Relative_Day'] = range(-WINDOW_BEFORE, len(window_df) - WINDOW_BEFORE)
    
    # Rebaser le prix à 100 au début de la fenêtre (J-5) pour comparer l'évolution
    start_price = window_df['Adj Close'].iloc[0]
    window_df['Normalized_Price'] = (window_df['Adj Close'] / start_price) * 100
    
    window_df['Event_ID'] = event_date_str # Pour distinguer les lignes sur le graph
    processed_events.append(window_df)

# Combiner tous les événements dans un seul DataFrame géant
all_events_df = pd.concat(processed_events)

# --- 3. VISUALISATION "WOW" ---
plt.figure(figsize=(12, 7))

# A. Tracer les lignes individuelles (le "bruit") en gris léger
sns.lineplot(
    data=all_events_df,
    x="Relative_Day",
    y="Normalized_Price",
    hue="Event_ID", # Une ligne par événement
    palette=["#bdc3c7"] * len(processed_events), # Couleur grise uniforme
    alpha=0.5, # Transparence
    legend=False,
    linewidth=1
)

# B. Tracer la moyenne et l'intervalle de confiance (le "signal") en couleur vive
sns.lineplot(
    data=all_events_df,
    x="Relative_Day",
    y="Normalized_Price",
    color="#c0392b", # Rouge foncé professionnel
    linewidth=3,
    label="Réaction Moyenne (±95% Confiance)"
)

# C. Elements graphiques pour la clarté
plt.axvline(x=0, color='black', linestyle='--', linewidth=1.5, label="Jour de l'Annonce (T=0)")
plt.axhline(y=100, color='grey', linestyle=':', linewidth=1)

plt.title(f"Impact Framework : Réaction Moyenne du {TICKER} aux Annonces CPI\n(Base 100 à T-{WINDOW_BEFORE})", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Jours Relatifs à l'Annonce", fontsize=12)
plt.ylabel("Performance Cumulée (Base 100)", fontsize=12)
plt.legend(loc='upper left', fontsize=10)

# Annotations pour les non-experts
plt.text(-WINDOW_BEFORE + 0.5, 99.5, "← Phase d'Anticipation", fontsize=10, color='grey')
plt.text(0.5, 100.2, "Phase de Digestion →", fontsize=10, color='grey')

plt.tight_layout()
print("Graphique généré avec succès.")
plt.show()
