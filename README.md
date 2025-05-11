# Steam Game Recommender
Pro funkci je nutné mít nainstalované následující balíčky:<br/>
Flask<br/>
python-dotenv<br/>
pandas<br/>
numpy<br/>
scikit-learn<br/>
steam-web-api<br/>
requests<br/>

Příklady Steam IDs na vyzkoušení:<br/>
Soukromý profil: 76561199176845392<br/>
Veřejné profily:<br/>
76561198058564905<br/>
76561198080947810<br/>
76561198053298539<br/>
<br/>

app.py obsahuje doporučovací logiku a na začátku bere autorův Steam API klíč ze souboru .env<br/>
steam_games.csv slouží jako lokální dataset se všemi kandidátskými hrami<br/>
