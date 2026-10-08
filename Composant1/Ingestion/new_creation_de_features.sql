CREATE TABLE player_features (
    -- Clés
    player_id               INTEGER REFERENCES players(player_id),
    saison                  VARCHAR(7),
    team_id                 INTEGER,
    feature_version         VARCHAR(10),  -- 'v1.0', 'v1.1'

    -- Contexte
    poss                    INTEGER,
    gp                      INTEGER,
    valuation_tier          VARCHAR(20),

    -- Profil physique
    height_inches           FLOAT,
    weight                  FLOAT,
    bmi                     FLOAT,
    age_at_date             FLOAT,
    peak_distance           FLOAT,
    saisons_experience      INTEGER,

    -- Stats shrinkées
    pts_per100_shrunk       FLOAT,
    reb_per100_shrunk       FLOAT,
    ast_per100_shrunk       FLOAT,
    stl_per100_shrunk       FLOAT,
    blk_per100_shrunk       FLOAT,
    tov_per100_shrunk       FLOAT,
    fgm_per100_shrunk       FLOAT,
    fga_per100_shrunk       FLOAT,
    fg3m_per100_shrunk      FLOAT,
    fg3a_per100_shrunk      FLOAT,
    ftm_per100_shrunk       FLOAT,
    fta_per100_shrunk       FLOAT,
    oreb_per100_shrunk      FLOAT,
    dreb_per100_shrunk      FLOAT,
    plus_minus_per100_shrunk FLOAT,
    blka_per100_shrunk      FLOAT,
    pf_per100_shrunk        FLOAT,

    -- Ratios avancés
    fg_pct                  FLOAT,
    fg3_pct                 FLOAT,
    ft_pct                  FLOAT,
    efg_pct                 FLOAT,
    ts_pct                  FLOAT,
    off_rating              FLOAT,
    def_rating              FLOAT,
    net_rating              FLOAT,
    usg_pct                 FLOAT,
    pie                     FLOAT,
    ast_pct                 FLOAT,
    oreb_pct                FLOAT,
    dreb_pct                FLOAT,
    reb_pct                 FLOAT,

    -- Features dérivées
    three_rate              FLOAT,
    free_throw_rate         FLOAT,
    turnover_rate           FLOAT,
    assist_rate             FLOAT,

    -- Features temporelles
    trend_pts               FLOAT,
    trend_pie               FLOAT,

    PRIMARY KEY (player_id, saison, team_id, feature_version)
);