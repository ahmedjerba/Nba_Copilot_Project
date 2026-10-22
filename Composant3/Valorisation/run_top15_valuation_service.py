import pandas as pd
from Composant3.Valorisation.model_registry import load_model
from Composant3.Valorisation.data_loader import load_pairs
from Composant3.Valorisation.feature_engineering import build_features

# 1. Chargement modèle production market_v2
payload = load_model('market_v2')
beta_model = payload['model']

# 2. Chargement données 2025-26
pairs = load_pairs(segment='all')
df_25 = pairs[pairs['saison'] == '2025-26'].copy()
print(f'Joueurs 2025-26 : {len(df_25)}')

# 3. Features
fe_25, _, _, _ = build_features(df_25, **payload['fe_flags'])
cap_2025 = 154647000.0

# 4. Évaluation via predict_player de valuation_service
results = []
for idx, row in fe_25.iterrows():
    p_df = fe_25.loc[[idx]]
    res = beta_model.predict_player(p_df, salary_cap=cap_2025)
    cap_hit = float(row['cap_hit']) if pd.notnull(row['cap_hit']) else 0.0
    val_dollars = res['valeur_dollars']
    surplus = val_dollars - cap_hit
    results.append({
        'player_id': int(row['player_id']),
        'player_name': row['player_name'],
        'team_name': row['team_name'],
        'position': row['position'],
        'age': round(float(row['age_at_date']), 1),
        'saisons_exp': int(row['saisons_experience']),
        'valeur_cap_pct': res['valeur_pct'],
        'valeur_dollars': val_dollars,
        'p10_dollars': res['p10_dollars'],
        'p90_dollars': res['p90_dollars'],
        'cap_hit_reel': cap_hit,
        'surplus': surplus
    })

res_df = pd.DataFrame(results)

# Sauvegarde pour persistance
res_df.to_csv('Composant3/Valorisation/predictions_valuation_service_2025_26.csv', index=False)

top15 = res_df.sort_values('valeur_dollars', ascending=False).head(15)

print('\n' + '='*110)
print('TOP 15 DES JOUEURS AYANT LA PLUS HAUTE VALEUR ESTIMÉE ($) - VALUATION SERVICE OFFICIEL (2025-2026)')
print('Salary Cap référence : 154,647,000 $')
print('='*110)
print(f"{'Rang':<4} {'Joueur':<24} {'Eq.':<4} {'Poste':<14} {'Exp':<4} {'Valeur ($)':<16} {'% Cap':<8} {'Salaire Réel':<16} {'Surplus ($)':<16} {'Statut'}")
print('-'*110)
for i, r in enumerate(top15.to_dict('records'), 1):
    statut = 'SOUS-PAYÉ 🟢' if r['surplus'] > 0 else 'SUR-PAYÉ 🔴'
    val_str = f"{r['valeur_dollars']:,} $"
    cap_str = f"{int(r['cap_hit_reel']):,} $"
    surp_str = f"{int(r['surplus']):+,} $"
    pct_str = f"{r['valeur_cap_pct']*100:.1f} %"
    print(f"{i:<4} {r['player_name']:<24} {str(r['team_name']):<4} {str(r['position']):<14} {r['saisons_exp']:<4} {val_str:<16} {pct_str:<8} {cap_str:<16} {surp_str:<16} {statut}")
print('='*110)
