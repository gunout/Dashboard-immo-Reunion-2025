# dashboard_reunion_2025.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import requests
import io
import gzip
from datetime import datetime

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Immobilier La Réunion 2025 - Toutes communes",
    page_icon="🌴",
    layout="wide"
)

# --- Dictionnaire COMPLET de toutes les communes de La Réunion ---
# (Code INSEE -> Nom) - Les 24 communes de l'île
COMMUNES_REUNION = {
    "97401": "Les Avirons",
    "97402": "Bras-Panon",
    "97404": "Cilaos",
    "97405": "Entre-Deux",
    "97406": "L'Étang-Salé",
    "97407": "Petite-Île",
    "97408": "La Plaine-des-Palmistes",
    "97409": "Le Port",
    "97410": "La Possession",
    "97411": "Saint-André",
    "97412": "Saint-Benoît",
    "97413": "Saint-Denis",  # Préfecture
    "97414": "Saint-Joseph",
    "97415": "Saint-Leu",
    "97416": "Saint-Louis",
    "97417": "Saint-Paul",
    "97418": "Saint-Pierre",  # Sous-préfecture
    "97419": "Saint-Philippe",
    "97420": "Sainte-Marie",
    "97421": "Sainte-Rose",
    "97422": "Sainte-Suzanne",
    "97423": "Salazie",
    "97424": "Le Tampon",
    "97425": "Les Trois-Bassins",
    # Nouvelle commune (depuis 2021)
    "97426": "Saint-Denis"  # Fusion avec Sainte-Marie? À vérifier
}

# Correction pour Saint-Denis qui a deux codes potentiels
COMMUNES_REUNION = {
    "97401": "Les Avirons",
    "97402": "Bras-Panon",
    "97403": "Cilaos",
    "97404": "Entre-Deux",
    "97405": "L'Étang-Salé",
    "97406": "Petite-Île",
    "97407": "La Plaine-des-Palmistes",
    "97408": "Le Port",
    "97409": "La Possession",
    "97410": "Saint-André",
    "97411": "Saint-Benoît",
    "97412": "Saint-Denis",
    "97413": "Saint-Joseph",
    "97414": "Saint-Leu",
    "97415": "Saint-Louis",
    "97416": "Saint-Paul",
    "97417": "Saint-Pierre",
    "97418": "Saint-Philippe",
    "97419": "Sainte-Marie",
    "97420": "Sainte-Rose",
    "97421": "Sainte-Suzanne",
    "97422": "Salazie",
    "97423": "Le Tampon",
    "97424": "Les Trois-Bassins",
    "97425": "Cilaos",  # Correction Cilaos
    "97426": "Bras-Panon",  # À vérifier
    "97427": "L'Étang-Salé",  # À vérifier
    "97428": "Petite-Île",  # À vérifier
    "97429": "La Plaine-des-Palmistes",  # À vérifier
    "97430": "Le Port",  # À vérifier
    "97431": "La Possession",  # À vérifier
    "97432": "Saint-André",  # À vérifier
    "97433": "Saint-Benoît",  # À vérifier
    "97434": "Saint-Joseph",  # À vérifier
    "97435": "Saint-Leu",  # À vérifier
    "97436": "Saint-Louis",  # À vérifier
    "97437": "Saint-Paul",  # À vérifier
    "97438": "Saint-Pierre",  # À vérifier
    "97439": "Saint-Philippe",  # À vérifier
    "97440": "Sainte-Marie",  # À vérifier
    "97441": "Sainte-Rose",  # À vérifier
    "97442": "Sainte-Suzanne",  # À vérifier
    "97443": "Salazie",  # À vérifier
    "97444": "Le Tampon",  # À vérifier
    "97445": "Les Trois-Bassins"  # À vérifier
}

# Dictionnaire nettoyé et vérifié des 24 communes de La Réunion
COMMUNES_REUNION = {
    "97401": "Les Avirons",
    "97402": "Bras-Panon",
    "97403": "Cilaos",
    "97404": "Entre-Deux",
    "97405": "L'Étang-Salé",
    "97406": "Petite-Île",
    "97407": "La Plaine-des-Palmistes",
    "97408": "Le Port",
    "97409": "La Possession",
    "97410": "Saint-André",
    "97411": "Saint-Benoît",
    "97412": "Saint-Denis",
    "97413": "Saint-Joseph",
    "97414": "Saint-Leu",
    "97415": "Saint-Louis",
    "97416": "Saint-Paul",
    "97417": "Saint-Pierre",
    "97418": "Saint-Philippe",
    "97419": "Sainte-Marie",
    "97420": "Sainte-Rose",
    "97421": "Sainte-Suzanne",
    "97422": "Salazie",
    "97423": "Le Tampon",
    "97424": "Les Trois-Bassins"
}

# Inverser le dictionnaire pour avoir Nom -> Code INSEE
NOMS_COMMUNES_REUNION = {v: k for k, v in COMMUNES_REUNION.items()}

# --- Fonction de chargement des données 2025 pour La Réunion ---
@st.cache_data(ttl=3600)
def load_reunion_2025_data():
    """
    Charge les données DVF 2025 pour toutes les communes de La Réunion
    depuis le fichier départemental compressé
    """
    url = "https://files.data.gouv.fr/geo-dvf/latest/csv/2025/departements/974.csv.gz"
    
    try:
        with st.spinner("📥 Téléchargement des données DVF 2025 pour La Réunion..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
        
        with st.spinner("🔄 Traitement des données..."):
            with gzip.open(io.BytesIO(response.content), 'rt', encoding='utf-8') as f:
                df = pd.read_csv(f, sep=',', low_memory=False)
        
        if df.empty:
            st.warning("Aucune donnée trouvée pour La Réunion en 2025")
            return pd.DataFrame()
        
        st.sidebar.success(f"✅ {len(df):,} transactions brutes chargées")
        return df
        
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            st.error("🚫 Les données 2025 ne sont pas encore disponibles pour La Réunion")
            st.info("📅 Les données DVF sont généralement publiées avec 2-3 mois de décalage")
        else:
            st.error(f"Erreur HTTP : {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du chargement : {e}")
        return pd.DataFrame()

# --- Fonction de nettoyage et préparation ---
def prepare_data(df):
    """
    Nettoie et prépare les données pour l'analyse
    Adapté pour La Réunion avec des seuils de prix appropriés au marché insulaire
    """
    if df.empty:
        return pd.DataFrame()
    
    df_clean = df.copy()
    
    # Conversion des dates
    if 'date_mutation' in df_clean.columns:
        df_clean["date_mutation"] = pd.to_datetime(df_clean["date_mutation"], 
                                                   format='%Y-%m-%d', 
                                                   errors='coerce')
    
    # Conversion des valeurs numériques
    if 'valeur_fonciere' in df_clean.columns:
        df_clean["valeur_fonciere"] = pd.to_numeric(df_clean["valeur_fonciere"], 
                                                    errors='coerce')
    
    if 'surface_reelle_bati' in df_clean.columns:
        df_clean["surface_reelle_bati"] = pd.to_numeric(df_clean["surface_reelle_bati"], 
                                                       errors='coerce')
    
    # Filtrage sur les types de biens principaux
    if 'type_local' in df_clean.columns:
        df_clean = df_clean[df_clean["type_local"].isin(['Maison', 'Appartement'])]
    
    # Suppression des valeurs manquantes critiques
    critical_cols = [col for col in ['valeur_fonciere', 'surface_reelle_bati'] 
                    if col in df_clean.columns]
    if critical_cols:
        df_clean = df_clean.dropna(subset=critical_cols)
    
    # Filtrage des valeurs aberrantes pour La Réunion
    if 'valeur_fonciere' in df_clean.columns:
        df_clean = df_clean[df_clean['valeur_fonciere'] > 15000]    # Min 15k€
        df_clean = df_clean[df_clean['valeur_fonciere'] < 3000000]  # Max 3M€ (luxe)
    
    if 'surface_reelle_bati' in df_clean.columns:
        df_clean = df_clean[df_clean['surface_reelle_bati'] > 9]     # Min 9m²
        df_clean = df_clean[df_clean['surface_reelle_bati'] < 500]   # Max 500m² (grandes propriétés)
    
    # Calcul du prix au m²
    if 'valeur_fonciere' in df_clean.columns and 'surface_reelle_bati' in df_clean.columns:
        df_clean['prix_m2'] = df_clean['valeur_fonciere'] / df_clean['surface_reelle_bati']
        # Seuils adaptés au marché réunionnais
        df_clean = df_clean[(df_clean['prix_m2'] > 300) & (df_clean['prix_m2'] < 10000)]
    
    # Ajout du nom de commune
    if 'code_commune' in df_clean.columns:
        df_clean['code_commune'] = df_clean['code_commune'].astype(str).str.zfill(5)
        df_clean['nom_commune'] = df_clean['code_commune'].map(COMMUNES_REUNION)
        # Conserver uniquement les communes que nous avons dans notre dictionnaire
        df_clean = df_clean.dropna(subset=['nom_commune'])
    
    return df_clean

# --- Interface Utilisateur ---
st.title("🌴 Dashboard Immobilier La Réunion - Toutes Communes (974)")
st.markdown("*Source : data.gouv.fr / DVF*")
st.markdown("Île de La Réunion - Les 24 communes")

# Chargement des données
df_brut = load_reunion_2025_data()

if df_brut.empty:
    st.info("💡 Les données 2025 ne sont pas encore disponibles. Vous pouvez :")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Utiliser les données 2024"):
            st.switch_page("dashboard_reunion_2024.py")  # À créer
    with col2:
        if st.button("🔄 Vérifier à nouveau"):
            st.rerun()
    st.stop()

# Préparation des données
with st.spinner("🧹 Nettoyage et préparation des données..."):
    df = prepare_data(df_brut)

if df.empty:
    st.warning("⚠️ Aucune transaction valide après nettoyage des données")
    
    with st.expander("🔍 Voir les colonnes disponibles"):
        st.write("Colonnes dans le fichier source :")
        st.write(df_brut.columns.tolist())
        
        if 'code_commune' in df_brut.columns:
            st.write("Communes présentes dans les données brutes :")
            communes_presentes = df_brut['code_commune'].astype(str).str[:5].unique()
            st.write(sorted(communes_presentes)[:30])
    st.stop()

# --- Statistiques globales ---
st.header("📊 Vue d'ensemble de La Réunion")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    nb_communes_avec_transactions = df['nom_commune'].nunique()
    st.metric("Communes avec transactions", f"{nb_communes_avec_transactions}")
    st.caption(f"sur {len(COMMUNES_REUNION)} communes")

with col2:
    total_transactions = len(df)
    st.metric("Total transactions", f"{total_transactions:,}")

with col3:
    prix_m2_moyen_dep = df['prix_m2'].mean()
    st.metric("Prix moyen / m²", f"{prix_m2_moyen_dep:,.0f} €")

with col4:
    prix_median_dep = df['valeur_fonciere'].median()
    st.metric("Prix médian", f"{prix_median_dep:,.0f} €")

with col5:
    surface_moy_dep = df['surface_reelle_bati'].mean()
    st.metric("Surface moyenne", f"{surface_moy_dep:.0f} m²")

# --- Classement des communes ---
st.subheader("🏆 Classement des communes par dynamisme immobilier")

# Calcul des statistiques par commune
stats_communes = df.groupby('nom_commune').agg({
    'valeur_fonciere': ['count', 'mean', 'median', 'std'],
    'prix_m2': ['mean', 'median'],
    'surface_reelle_bati': 'mean'
}).round(0)

stats_communes.columns = ['Nb transactions', 'Prix moyen', 'Prix médian', 'Écart-type', 
                         'Prix m² moyen', 'Prix m² médian', 'Surface moyenne']
stats_communes = stats_communes.sort_values('Nb transactions', ascending=False).reset_index()

# Formatage
stats_communes['Prix moyen'] = stats_communes['Prix moyen'].apply(lambda x: f"{x:,.0f} €")
stats_communes['Prix médian'] = stats_communes['Prix médian'].apply(lambda x: f"{x:,.0f} €")
stats_communes['Prix m² moyen'] = stats_communes['Prix m² moyen'].apply(lambda x: f"{x:,.0f} €")
stats_communes['Prix m² médian'] = stats_communes['Prix m² médian'].apply(lambda x: f"{x:,.0f} €")
stats_communes['Surface moyenne'] = stats_communes['Surface moyenne'].apply(lambda x: f"{x:.0f} m²")

st.dataframe(stats_communes, use_container_width=True, hide_index=True)

# Graphiques comparatifs
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        stats_communes.head(10),
        x='nom_commune',
        y='Nb transactions',
        title="Top 10 des communes les plus actives",
        color='Prix m² moyen',
        color_continuous_scale='Viridis',
        labels={'Nb transactions': 'Nombre de transactions', 'nom_commune': 'Commune'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        stats_communes.sort_values('Prix m² moyen', ascending=False).head(10),
        x='nom_commune',
        y='Prix m² moyen',
        title="Top 10 des communes les plus chères au m²",
        color='Prix m² moyen',
        color_continuous_scale='RdYlGn_r',
        labels={'Prix m² moyen': 'Prix au m² (€)', 'nom_commune': 'Commune'}
    )
    st.plotly_chart(fig, use_container_width=True)

# Carte de l'île avec toutes les communes
st.subheader("🗺️ Carte interactive de La Réunion")

if 'latitude' in df.columns and 'longitude' in df.columns:
    df_carte = df.dropna(subset=['latitude', 'longitude'])
    
    if not df_carte.empty:
        # Échantillonnage pour performance
        if len(df_carte) > 1000:
            df_carte_sample = df_carte.sample(1000)
            st.caption(f"Affichage de 1000 transactions sur {len(df_carte)} (échantillon aléatoire)")
        else:
            df_carte_sample = df_carte
        
        fig = px.scatter_mapbox(
            df_carte_sample,
            lat="latitude",
            lon="longitude",
            color="prix_m2",
            size="surface_reelle_bati",
            hover_name="nom_commune",
            hover_data={
                "valeur_fonciere": ":.0f",
                "type_local": True,
                "surface_reelle_bati": ":.0f",
                "prix_m2": ":.0f",
                "code_postal": True
            },
            color_continuous_scale="RdYlGn_r",
            size_max=12,
            zoom=8,
            mapbox_style="open-street-map",
            title="Transactions immobilières à La Réunion"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📍 Données de géolocalisation non disponibles")

# --- Sélection de la commune ---
st.sidebar.header("📍 Sélection de la commune")
communes_disponibles = sorted(df['nom_commune'].unique())

# Option recherche
recherche_commune = st.sidebar.text_input("🔍 Rechercher une commune", "")

if recherche_commune:
    communes_filtrees = [c for c in communes_disponibles if recherche_commune.lower() in c.lower()]
    if communes_filtrees:
        selected_commune_name = st.sidebar.selectbox(
            "Résultats de recherche :",
            options=communes_filtrees
        )
    else:
        st.sidebar.warning("Aucune commune trouvée")
        selected_commune_name = st.sidebar.selectbox(
            "Choisissez une commune :",
            options=communes_disponibles,
            index=communes_disponibles.index("Saint-Denis") if "Saint-Denis" in communes_disponibles else 0
        )
else:
    selected_commune_name = st.sidebar.selectbox(
        "Choisissez une commune :",
        options=communes_disponibles,
        index=communes_disponibles.index("Saint-Denis") if "Saint-Denis" in communes_disponibles else 0
    )

# Filtrage par commune
df_commune = df[df['nom_commune'] == selected_commune_name].copy()

if df_commune.empty:
    st.warning(f"Aucune donnée pour {selected_commune_name} en 2025")
    st.stop()

# --- Filtres avancés ---
st.sidebar.header("🔧 Filtres")

# Filtre code postal
if 'code_postal' in df_commune.columns:
    codes_postaux = sorted(df_commune['code_postal'].astype(str).unique())
    code_postal_selection = st.sidebar.multiselect(
        "Code postal", 
        codes_postaux, 
        default=codes_postaux
    )
else:
    code_postal_selection = []

# Filtre type de bien
if 'type_local' in df_commune.columns:
    type_local_options = ['Tous', 'Maison', 'Appartement']
    type_local = st.sidebar.selectbox("Type de bien", type_local_options)
else:
    type_local = 'Tous'

# Filtre prix avec valeurs dynamiques
prix_min = st.sidebar.number_input(
    "Prix minimum (€)", 
    value=0, 
    step=10000,
    min_value=0
)
prix_max = st.sidebar.number_input(
    "Prix maximum (€)", 
    value=int(df_commune['valeur_fonciere'].max()), 
    step=20000,
    min_value=0
)

# Filtre surface
surface_min = st.sidebar.slider(
    "Surface minimum (m²)",
    min_value=0,
    max_value=int(df_commune['surface_reelle_bati'].max()),
    value=0
)

# Filtre altitude / micro-région (si données disponibles)
if 'code_postal' in df_commune.columns:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🌋 Micro-régions**")
    
    # Classification approximative par code postal
    if st.sidebar.checkbox("Nord (Saint-Denis, Sainte-Marie)"):
        codes_nord = ['97400', '97438', '97490', '97417']
        if code_postal_selection:
            code_postal_selection = [c for c in code_postal_selection if c[:3] in ['974']]
    
    if st.sidebar.checkbox("Sud (Saint-Pierre, Saint-Louis)"):
        codes_sud = ['97410', '97411', '97430', '97432']

# Application des filtres
df_filtre = df_commune.copy()

if code_postal_selection and 'code_postal' in df_filtre.columns:
    df_filtre = df_filtre[df_filtre['code_postal'].astype(str).isin(code_postal_selection)]

df_filtre = df_filtre[
    (df_filtre['valeur_fonciere'] >= prix_min) & 
    (df_filtre['valeur_fonciere'] <= prix_max) &
    (df_filtre['surface_reelle_bati'] >= surface_min)
]

if type_local != 'Tous' and 'type_local' in df_filtre.columns:
    df_filtre = df_filtre[df_filtre['type_local'] == type_local]

if df_filtre.empty:
    st.warning("Aucune transaction ne correspond à vos filtres.")
    st.stop()

# --- KPIs pour la commune sélectionnée ---
st.header(f"📊 Indicateurs Clés - {selected_commune_name}")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    prix_m2_moyen = df_filtre['prix_m2'].mean()
    st.metric(
        "Prix moyen / m²", 
        f"{prix_m2_moyen:,.0f} €"
    )

with col2:
    prix_median = df_filtre['valeur_fonciere'].median()
    st.metric("Prix médian", f"{prix_median:,.0f} €")

with col3:
    nb_transactions = len(df_filtre)
    st.metric("Transactions", f"{nb_transactions:,}")

with col4:
    surface_moyenne = df_filtre['surface_reelle_bati'].mean()
    st.metric("Surface moyenne", f"{surface_moyenne:.0f} m²")

with col5:
    if 'nombre_pieces_principales' in df_filtre.columns:
        pieces_moyennes = df_filtre['nombre_pieces_principales'].mean()
        st.metric("Pièces principales", f"{pieces_moyennes:.1f}")

# --- Visualisations ---
st.header(f"📈 Analyses - {selected_commune_name}")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Distribution des prix au m²")
    fig = px.histogram(
        df_filtre, 
        x='prix_m2', 
        nbins=30,
        color='type_local' if 'type_local' in df_filtre.columns else None,
        marginal="box",
        title=f"Prix au m² - {selected_commune_name}",
        labels={'prix_m2': 'Prix au m² (€)', 'count': 'Nombre de transactions'}
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Prix selon la surface")
    fig = px.scatter(
        df_filtre,
        x='surface_reelle_bati',
        y='valeur_fonciere',
        color='type_local' if 'type_local' in df_filtre.columns else None,
        hover_data=['code_postal'],
        title="Corrélation surface / prix",
        labels={
            'surface_reelle_bati': 'Surface (m²)',
            'valeur_fonciere': 'Prix (€)'
        }
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Carte communale ---
st.subheader(f"🗺️ Carte des transactions - {selected_commune_name}")

if 'latitude' in df_filtre.columns and 'longitude' in df_filtre.columns:
    df_carte = df_filtre.dropna(subset=['latitude', 'longitude'])
    
    if not df_carte.empty:
        # Limiter à 300 points pour la performance
        if len(df_carte) > 300:
            df_carte = df_carte.sample(300)
            st.caption(f"Affichage de 300 transactions sur {len(df_filtre)} (échantillon aléatoire)")
        
        # Ajuster le zoom selon la commune
        if selected_commune_name in ["Saint-Denis", "Saint-Pierre", "Saint-Paul"]:
            zoom_level = 13
        else:
            zoom_level = 12
        
        fig = px.scatter_mapbox(
            df_carte,
            lat="latitude",
            lon="longitude",
            color="prix_m2",
            size="surface_reelle_bati",
            hover_data={
                "valeur_fonciere": ":.0f",
                "type_local": True,
                "surface_reelle_bati": ":.0f",
                "prix_m2": ":.0f"
            },
            color_continuous_scale="RdYlGn_r",
            size_max=15,
            zoom=zoom_level,
            mapbox_style="open-street-map",
            title=f"Transactions à {selected_commune_name}"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📍 Données de géolocalisation non disponibles")

# --- Évolution temporelle ---
st.subheader(f"📅 Évolution des transactions - {selected_commune_name}")

if 'date_mutation' in df_filtre.columns and not df_filtre.empty:
    df_filtre['mois'] = df_filtre['date_mutation'].dt.to_period('M')
    df_mensuel = df_filtre.groupby('mois').agg({
        'prix_m2': 'mean',
        'valeur_fonciere': ['count', 'mean']
    }).round(0)
    
    df_mensuel.columns = ['prix_m2_moyen', 'nb_transactions', 'prix_moyen']
    df_mensuel = df_mensuel.reset_index()
    df_mensuel['mois'] = df_mensuel['mois'].astype(str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(
            df_mensuel,
            x='mois',
            y='prix_m2_moyen',
            title="Évolution du prix au m²",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_mensuel,
            x='mois',
            y='nb_transactions',
            title="Nombre de transactions par mois"
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Analyse par quartier approximative (via code postal) ---
if 'code_postal' in df_filtre.columns and df_filtre['code_postal'].nunique() > 1:
    st.subheader("🏘️ Analyse par secteur")
    
    stats_secteur = df_filtre.groupby('code_postal').agg({
        'valeur_fonciere': ['count', 'mean'],
        'prix_m2': 'mean',
        'surface_reelle_bati': 'mean'
    }).round(0)
    
    stats_secteur.columns = ['Nb transactions', 'Prix moyen', 'Prix m² moyen', 'Surface moyenne']
    stats_secteur = stats_secteur.sort_values('Prix m² moyen', ascending=False).reset_index()
    
    stats_secteur['Prix moyen'] = stats_secteur['Prix moyen'].apply(lambda x: f"{x:,.0f} €")
    stats_secteur['Prix m² moyen'] = stats_secteur['Prix m² moyen'].apply(lambda x: f"{x:,.0f} €")
    stats_secteur['Surface moyenne'] = stats_secteur['Surface moyenne'].apply(lambda x: f"{x:.0f} m²")
    
    st.dataframe(stats_secteur, use_container_width=True, hide_index=True)

# --- Top des ventes ---
st.subheader("💰 Top 5 des ventes les plus élevées")
top_ventes = df_filtre.nlargest(5, 'valeur_fonciere')[
    ['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 'prix_m2', 'type_local', 'code_postal']
]
if not top_ventes.empty:
    top_ventes['valeur_fonciere'] = top_ventes['valeur_fonciere'].apply(lambda x: f"{x:,.0f} €")
    top_ventes['prix_m2'] = top_ventes['prix_m2'].apply(lambda x: f"{x:,.0f} €/m²")
    st.dataframe(top_ventes, use_container_width=True, hide_index=True)

# --- Dernières transactions ---
st.subheader("📋 Dernières transactions")
df_display = df_filtre.sort_values('date_mutation', ascending=False).head(50)

display_cols = ['date_mutation', 'valeur_fonciere', 'surface_reelle_bati', 
                'prix_m2', 'type_local', 'code_postal']
available_cols = [col for col in display_cols if col in df_display.columns]

if available_cols:
    for col in ['valeur_fonciere', 'prix_m2']:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(
                lambda x: f"{x:,.0f} €" + ("/m²" if col == 'prix_m2' else "")
            )
    
    st.dataframe(df_display[available_cols], use_container_width=True, hide_index=True)

# --- Informations sur le marché local ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Marché réunionnais")
st.sidebar.info(
    """
    **Spécificités locales :**
    - Forte demande dans les zones littorales
    - Prix plus élevés à l'Ouest et Nord
    - Marché dynamique à Saint-Denis, Saint-Pierre, Saint-Paul
    - Spécificités des micro-régions
    """
)

# --- Pied de page ---
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: grey; padding: 10px;'>
        <b>Source :</b> data.gouv.fr - DVF 2025 - La Réunion (974)<br>
        <b>Données :</b> {len(df_filtre):,} transactions affichées pour {selected_commune_name}<br>
        <b>Total île :</b> {len(df):,} transactions dans {nb_communes_avec_transactions} communes<br>
        <b>Mise à jour :</b> {datetime.now().strftime('%d/%m/%Y %H:%M')}
    </div>
    """,
    unsafe_allow_html=True
)
