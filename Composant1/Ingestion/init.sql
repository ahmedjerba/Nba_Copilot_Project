CREATE TABLE teams (
    team_id         INTEGER PRIMARY KEY,
    team_name       VARCHAR(50) NOT NULL,
    abbreviation    VARCHAR(5),
    conference      VARCHAR(4),
    division        VARCHAR(20)
);

CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,
    player_name     VARCHAR(100) NOT NULL,
    birthdate       DATE,
    position        VARCHAR(50),
    height          VARCHAR(10),
    weight          FLOAT,
    country         VARCHAR(50),
    draft_year      INTEGER,
    draft_round     INTEGER,
    draft_number    INTEGER,
    team_id         INTEGER,
    team_name       VARCHAR(50),
    spotrac_id      INTEGER UNIQUE
);

CREATE TABLE player_id_map (
    player_id           INTEGER PRIMARY KEY REFERENCES players(player_id),
    spotrac_player_id   INTEGER,
    player_name_stats   VARCHAR(100),
    player_name_spotrac VARCHAR(100),
    match_type          VARCHAR(20)
);

CREATE TABLE player_stats (
    player_id           INTEGER REFERENCES players(player_id),
    saison              VARCHAR(7),
    team_id             INTEGER,
    team_abbreviation   VARCHAR(5),
    team_count          INTEGER,
    age                 FLOAT,
    gp                  INTEGER,
    w_pct               FLOAT,
    poss                INTEGER,
    valuation_tier      VARCHAR(20),
    -- per 100 possessions
    pts_per100          FLOAT,
    reb_per100          FLOAT,
    ast_per100          FLOAT,
    stl_per100          FLOAT,
    blk_per100          FLOAT,
    blka_per100         FLOAT,
    tov_per100          FLOAT,
    pf_per100           FLOAT,
    fgm_per100          FLOAT,
    fga_per100          FLOAT,
    fg3m_per100         FLOAT,
    fg3a_per100         FLOAT,
    ftm_per100          FLOAT,
    fta_per100          FLOAT,
    oreb_per100         FLOAT,
    dreb_per100         FLOAT,
    plus_minus_per100   FLOAT,
    -- ratios avancés
    fg_pct              FLOAT,
    fg3_pct             FLOAT,
    ft_pct              FLOAT,
    efg_pct             FLOAT,
    ts_pct              FLOAT,
    off_rating          FLOAT,
    def_rating          FLOAT,
    net_rating          FLOAT,
    pace                FLOAT,
    usg_pct             FLOAT,
    pie                 FLOAT,
    ast_pct             FLOAT,
    ast_to              FLOAT,
    ast_ratio           FLOAT,
    oreb_pct            FLOAT,
    dreb_pct            FLOAT,
    reb_pct             FLOAT,
    tm_tov_pct          FLOAT,
    fgm_adv             FLOAT,
    fga_adv             FLOAT,
    fg_pct_adv          FLOAT,
    -- per game (affichage uniquement)
    min_pg              FLOAT,
    pts_pg              FLOAT,
    reb_pg              FLOAT,
    ast_pg              FLOAT,
    stl_pg              FLOAT,
    blk_pg              FLOAT,
    tov_pg              FLOAT,
    fgm_pg              FLOAT,
    fga_pg              FLOAT,
    fg3m_pg             FLOAT,
    fg3a_pg             FLOAT,
    ftm_pg              FLOAT,
    fta_pg              FLOAT,
    oreb_pg             FLOAT,
    dreb_pg             FLOAT,
    plus_minus_pg       FLOAT,
    PRIMARY KEY (player_id, saison, team_id)
);

CREATE TABLE contracts (
    player_id           INTEGER REFERENCES players(player_id),
    saison              VARCHAR(7),
    spotrac_player_id   INTEGER,
    base_salary         BIGINT,
    cap_hit             BIGINT,
    years_remaining     INTEGER,
    contract_value      BIGINT,
    PRIMARY KEY (player_id, saison)
);

CREATE TABLE team_payrolls (
    team_id             INTEGER REFERENCES teams(team_id),
    saison              VARCHAR(7),
    total_cap_hit       BIGINT,
    total_base_salary   BIGINT,
    nb_joueurs          INTEGER,
    PRIMARY KEY (team_id, saison)
);

CREATE TABLE risk_flags (
    id              SERIAL PRIMARY KEY,
    player_id       INTEGER REFERENCES players(player_id),
    flag_type       VARCHAR(20),
    description     TEXT,
    source_url      TEXT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE watchlist_entries (
    id          SERIAL PRIMARY KEY,
    player_id   INTEGER REFERENCES players(player_id),
    user_team   INTEGER REFERENCES teams(team_id),
    added_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE alerts (
    id          SERIAL PRIMARY KEY,
    player_id   INTEGER REFERENCES players(player_id),
    alert_type  VARCHAR(30),
    message     TEXT,
    seen        BOOLEAN DEFAULT FALSE,
    created_at  TIMESTAMP DEFAULT NOW()
);