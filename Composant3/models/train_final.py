import joblib
import os
from Composant3.data.load import load_training_pairs
from Composant3.models import ridge, quantile

# Charger toutes les données
df = load_training_pairs()

# Entraîner sur toutes les saisons (train final)
train_all = df[df["season_year"].isin([2021, 2022, 2023, 2024])]

# Ridge final
model_final = ridge.train(train_all)

# Quantiles finaux
models_q_final = quantile.train(train_all)

# Sauvegarder
os.makedirs("models", exist_ok=True)
joblib.dump(model_final,    "models/ridge_pie_final.pkl")
joblib.dump(models_q_final, "models/quantile_final.pkl")

print("Modèles sauvegardés ✅")
print(f"  Ridge best alpha : {model_final.best_params_['model__alpha']}")