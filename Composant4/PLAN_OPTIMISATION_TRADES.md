# Plan d'Optimisation du Composant 4 — Moteur CBA & Recherche Opérationnelle

Ce document détaille les anomalies identifiées, la modélisation mathématique retenue et le plan d'implémentation pour rendre le moteur CBA conforme à la réalité d'un front-office NBA.

---

## 1. Diagnostics & Anomalies Résolues

### Anomalie A : Packages multi-équipes illogiques (ex: Embiid [PHI] + Booker [PHX])
- **Cause** : Le pool `available` réunit les joueurs des 29 autres franchises sans contrainte de regroupement par franchise dans le modèle CP-SAT.
- **Solution** : 
  1. Ajouter une variable binaire d'activation de franchise partenaire $z_t \in \{0, 1\}$ pour chaque équipe adverse $t$.
  2. Imposer $\sum_{t} z_t = 1$ (trade bilatéral classique).
  3. Lier chaque joueur entrant $j$ à son équipe : $y_j \le z_{\text{team}(j)}$.
  4. Ainsi, tous les joueurs entrants proviennent obligatoirement et exclusivement de la même franchise.

### Anomalie B : "Waive" (libération) systématiquement en tête du classement
- **Cause** : Un joueur surpayé a un surplus de valeur négatif ($surplus = valeur\_terrain - cap\_hit < 0$, ex: $-14.5\text{M}\$$ pour Murray). Dans la formule actuelle :
  $$value\_delta = estimated\_value\_in - estimated\_value\_out = 0 - (-14.5\text{M}\$) = +14.5\text{M}\$$
  Le moteur interprétait la perte d'un joueur sous contrat garanti comme un gain net sans coût financier.
- **Réalité CBA** : Couper un contrat garanti laisse 100% du salaire en "dead money" sur le payroll (ou étalé via la *stretch provision*). On perd la production terrain pour $0\$$ d'économie immédiate, sauf négociation de buyout ou claim aux waivers.
- **Solution** :
  1. Réviser la fonction d'utilité / `value_delta` pour les waives : la perte sportive est comptabilisée, et le soulagement salarial n'existe que s'il y a une économie de taxe réelle ou de cap réel.
  2. Séparer l'évaluation des waives et des trades dans le rapport de l'engine afin que des coupes d'effectif ne viennent pas masquer les meilleures opportunités de trades.

### Anomalie C : Flux de travail inversé (besoin du mode "Target Player")
- **Cause** : Le moteur actuel ne fonctionnait qu'en mode "Shotgun" (exploration combinatoire aveugle sur 50 joueurs aléatoires).
- **Réalité Front-Office** : Un GM identifie une cible (un joueur ou un profil) et cherche quel package permet de l'obtenir selon le CBA.
- **Solution** :
  1. Implémenter `propose_trade_for_target(team_id, target_player_id, ...)` :
     - Fixe le joueur cible dans le trade ($y_{\text{cible}} = 1$).
     - Identifie l'équipe propriétaire de la cible.
     - Utilise CP-SAT pour trouver le package optimal de joueurs sortants (et éventuels joueurs de complément / fillers de l'équipe adverse) qui satisfait le salary matching CBA.
  2. Implémenter `propose_trades_with_team(team_id, partner_team_id, ...)` pour négocier franchise à franchise.

---

## 2. Plan d'Implémentation Étape par Étape

1. **[`Composant4/optimizer.py`](file:///c:/Users/user/Desktop/nba_copilot/Composant4/optimizer.py)** :
   - Mise à jour du solveur CP-SAT avec la contrainte d'unicité de la franchise partenaire ($y_j \le z_t$, $\sum z_t \le 1$).
   - Ajout du support de cible forcée : `target_player_ids` ou `target_team_id`.
   - Fonction dédiée `optimize_target_trade(roster, target_player, partner_roster, state, rules, saison)`.

2. **[`Composant4/schemas.py`](file:///c:/Users/user/Desktop/nba_copilot/Composant4/schemas.py)** & **[`Composant4/engine.py`](file:///c:/Users/user/Desktop/nba_copilot/Composant4/engine.py)** :
   - Rectification du calcul de rentabilité pour `MoveType.WAIVE` (prise en compte du dead cap garanti).
   - Ajout des fonctions d'API publique :
     - `propose_trade_for_target(team_id, target_player_id, ...)`
     - `propose_trades_with_team(team_id, partner_team_id, ...)`
   - Organisation du rapport `EngineResult` avec section trades ciblés et classement cohérent.

3. **Validation & Tests** :
   - Enrichissement de `test_composant4.py` pour valider :
     - Que tous les joueurs reçus dans chaque trade proviennent de la même franchise.
     - Que le mode cible fonctionne avec un joueur précis (ex: trouver le package pour acquérir Mikal Bridges ou un joueur du marché).
     - Que les waives ne trustent plus la tête du classement de manière absurde.
