import os
import time
import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from dotenv import load_dotenv
from steam_web_api import Steam
from sklearn.metrics.pairwise import cosine_similarity
import requests

# Vytvoření Flask aplikace
app = Flask(__name__)

# Načtení env variable a Steam API klíče
load_dotenv()
STEAM_API_KEY = os.environ.get("STEAM_API_KEY")
steam = Steam(STEAM_API_KEY)

# Cache pro owned_games
USER_CACHE = {} # Slovník
CACHE_TIME = 5 * 60  # Definování doby cache, 5 minut
MIN_GAMES = 5

# Vrací cached odpověď API pro dané Steam ID, 
# případně udělá nové volání a uloží výsledek do cache
def get_cached_games(steam_id):
    now = time.time()
    entry = USER_CACHE.get(steam_id)
    if entry and now - entry["timestamp"] < CACHE_TIME:
        return entry["response"]
    try:
        response = steam.users.get_owned_games(steam_id)
    except requests.exceptions.RequestException:
        raise
    USER_CACHE[steam_id] = { 
        "response": response,
        "timestamp": now
    }
    return response

# Načti csv soubor se všema hrama
CANDIDATE_CSV = "steam_games.csv"
candidate_df = pd.read_csv(CANDIDATE_CSV)

# Výpis používaných sloupců
used_candidate_columns = [
    "url",
    "name",
    "popular_tags"
]
candidate_df = candidate_df[used_candidate_columns]

# Získání appid z URL Steam store.
def extract_appid(url):
    try:
        parts = url.split("/")
        index_appid = parts.index("app")
        return int(parts[index_appid + 1]) # Získám to, co je po /app/TADY
    except Exception:
        return None
    
candidate_df["appid"] = candidate_df["url"].apply(extract_appid)
candidate_df["appid"] = candidate_df["appid"].astype("Int64")

# Získej URL header obrázku pro hru, změň na int pro smazání desetinné čárky
def make_header_url(appid):
    if pd.isna(appid):
        return ""
    try:
        appid2 = int(appid)
        return (f"https://cdn.akamai.steamstatic.com/steam/"
                f"apps/{appid2}/header.jpg")
    except (ValueError, TypeError):
        return ""

candidate_df["header_image_url"] = candidate_df["appid"].apply(make_header_url)

# Rozdělení řetězce tagů dle čárek
# Check, zdali je tag_str = String, potom rozdělim a lowercase a smažu bílé znaky
def extract_tags(tag_string):
    if isinstance(tag_string, str):
        return [t.strip().lower() for t in tag_string.split(",")]
    return []

candidate_df["tags_list"] = candidate_df["popular_tags"].apply(extract_tags)

# Udělam seznam všech tagů
def build_tag_vocab(user_df, candidate_df):
    user_tags = set()
    for tags in user_df["tags_list"]:
        user_tags.update(tags)
    candidate_tags = set()
    for tags in candidate_df["tags_list"]:
        candidate_tags.update(tags)
    return sorted(user_tags.union(candidate_tags))

# all_tags = ['action', 'rpg'] a game_tags = ['rpg'] vyjde vector = array([0., 1.])
def create_vector(game_tags, all_tags):
    vector = np.zeros(len(all_tags))
    for i, tag in enumerate(all_tags):
        if tag in game_tags:
            vector[i] = 1
    return vector

def recommend_games(steam_id):
    try:
        user_details = steam.users.get_user_details(steam_id=steam_id)
    except Exception:
        return None, ("Chyba při komunikaci se Steam API.")
    
    player = user_details.get("player", {})
    if not player:
        return None, ("Uživatel nebyl nalezen. Zkontrolujte Steam ID.")
    # Communityvisibilitystate -> 3 = veřejný, 1 nebo 2 soukromý
    if player.get("communityvisibilitystate") != 3:
        return None, ("Váš profil je nastavený jako soukromý. V nastavení Steam změňte profil na veřejný.")
    
    # Získání her uživatele (z cache nebo API)
    try:
        user_games_response = get_cached_games(steam_id)
    except Exception:
        return None, ("Chyba při komunikaci se Steam API.")
    
    games_data = user_games_response.get("games")
    if not games_data:
        return None, ("Nebyly nalezeny žádné hry na Vašem profilu.")
    
    if len(games_data) < MIN_GAMES:
        return None, (f"Potřebujete mít alespoň {MIN_GAMES} her.")

    df_games = pd.DataFrame(games_data)
    
    used_user_column = [
    "appid",
    "name",
    "playtime_forever"
    ]
    
    df_games = df_games[used_user_column]

   # Spoj s tagama z csv souboru
    df_games = pd.merge(
        df_games,
        candidate_df[["appid", "tags_list", "header_image_url", "url"]],
        on="appid",
        how="left"
    )
    df_games["header_image_url"] = df_games["header_image_url"].fillna("")
    
    # Checkujeme, jestli je list tagů opravdu "list"
    def list_for_sure(x):
        if isinstance(x, list):
            return x
        else:
            return []
    
    df_games["tags_list"] = df_games["tags_list"].apply(list_for_sure)

    all_tags = build_tag_vocab(df_games, candidate_df)
    
    user_X = np.array([
        create_vector(tags, all_tags)
        for tags in df_games["tags_list"]
    ])

    # Vytvoř uživatelský profil na základě playtimu jako vah
    playtimes = df_games["playtime_forever"].values
    if playtimes.sum() > 0:
        user_profile = np.average(user_X, axis=0, weights=playtimes)
    else:
        user_profile = np.mean(user_X, axis=0)

    # Vyfiltruj z csv souboru hry, které uživatel již vlastní
    owned_games = set(df_games["appid"])
    candidate_df_filtered = candidate_df[~candidate_df["appid"].isin(owned_games)].copy()
    
    X_candidates = np.array([
        create_vector(tags, all_tags)
        for tags in candidate_df_filtered["tags_list"]
    ])
    
    if X_candidates.size == 0:
        return None, "Nepodařilo se nalézt žádné kandidátské hry."

    # Vypočítej kosinovou podobnost mezi uživatelským profilem a kandidáty
    similarities = cosine_similarity(
        user_profile.reshape(1, -1),
        X_candidates
    ).flatten()

    # Vyber nejlepších n doporučeních
    top_n = 10

    recommended_indexes = similarities.argsort()[-top_n:][::-1]
    recommended_games = candidate_df_filtered.iloc[recommended_indexes].copy()
    recommended_games["header_image_url"] = recommended_games["header_image_url"].fillna("")

    return recommended_games, None

# Flask
@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = None
    error_message = None
    if request.method == "POST":
        steam_id = request.form.get("steam_id", "").strip()
        if not steam_id:
            error_message = "Prosím, zadejte Steam ID."
        else:
            try:
                recs, err = recommend_games(steam_id)
            except Exception:
                error_message = ("Chyba při volání Steam API.")
            else:
                if err:
                    error_message = err
                else:
                    recommendations = recs.to_dict("records")
    return render_template("index.html",
                           recommendations=recommendations,
                           error=error_message)

if __name__ == "__main__":
    app.run(debug=True)