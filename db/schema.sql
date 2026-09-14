-- Edge project schema
-- Run via: psql -h localhost -U <user> -d edge -f db/schema.sql

-- ============================================================
-- Teams
-- ============================================================
CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY,    -- API-Football team ID
    name        TEXT    NOT NULL,
    league      TEXT    NOT NULL DEFAULT 'EPL',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Matches
-- ============================================================
CREATE TABLE IF NOT EXISTS matches (
    match_id        INTEGER PRIMARY KEY,   -- API-Football fixture ID
    date            TIMESTAMPTZ NOT NULL,
    season          INTEGER NOT NULL,
    matchday        INTEGER,
    home_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id    INTEGER NOT NULL REFERENCES teams(team_id),
    home_goals      INTEGER,
    away_goals      INTEGER,
    result          CHAR(1) CHECK (result IN ('H', 'D', 'A')),  -- H=home win, D=draw, A=away win
    status          TEXT NOT NULL DEFAULT 'NS',                  -- NS=not started, FT=full time, etc.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_matches_season ON matches(season);
CREATE INDEX IF NOT EXISTS idx_matches_date   ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_home   ON matches(home_team_id);
CREATE INDEX IF NOT EXISTS idx_matches_away   ON matches(away_team_id);

-- ============================================================
-- Players
-- ============================================================
CREATE TABLE IF NOT EXISTS players (
    player_id   INTEGER PRIMARY KEY,   -- API-Football player ID
    name        TEXT    NOT NULL,
    position    TEXT,                  -- Goalkeeper, Defender, Midfielder, Attacker
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ============================================================
-- Team stats per match (one row per team per match)
-- ============================================================
CREATE TABLE IF NOT EXISTS team_stats (
    id              SERIAL PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    xg              NUMERIC(5, 2),
    shots_total     INTEGER,
    shots_on_target INTEGER,
    possession_pct  NUMERIC(5, 2),
    passes_total    INTEGER,
    pass_accuracy   NUMERIC(5, 2),
    corners         INTEGER,
    fouls           INTEGER,
    yellow_cards    INTEGER,
    red_cards       INTEGER,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_team_stats_match  ON team_stats(match_id);
CREATE INDEX IF NOT EXISTS idx_team_stats_team   ON team_stats(team_id);

-- ============================================================
-- Lineups (one row per team per match)
-- ============================================================
CREATE TABLE IF NOT EXISTS lineups (
    lineup_id           SERIAL PRIMARY KEY,
    match_id            INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id             INTEGER NOT NULL REFERENCES teams(team_id),
    formation           TEXT,           -- e.g. "4-3-3"
    release_timestamp   TIMESTAMPTZ,    -- when the lineup was confirmed/released
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_lineups_match ON lineups(match_id);

-- ============================================================
-- Lineup players (starters + subs per lineup)
-- ============================================================
CREATE TABLE IF NOT EXISTS lineup_players (
    id              SERIAL PRIMARY KEY,
    lineup_id       INTEGER NOT NULL REFERENCES lineups(lineup_id) ON DELETE CASCADE,
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    is_starter      BOOLEAN NOT NULL DEFAULT TRUE,
    jersey_number   INTEGER,
    position        TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (lineup_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_lineup_players_lineup  ON lineup_players(lineup_id);
CREATE INDEX IF NOT EXISTS idx_lineup_players_player  ON lineup_players(player_id);

-- ============================================================
-- Player stats per match
-- ============================================================
CREATE TABLE IF NOT EXISTS player_stats (
    id              SERIAL PRIMARY KEY,
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    match_id        INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    team_id         INTEGER NOT NULL REFERENCES teams(team_id),
    minutes_played  INTEGER,
    rating          NUMERIC(4, 2),     -- API-Football player rating (e.g. 7.8)
    goals           INTEGER DEFAULT 0,
    assists         INTEGER DEFAULT 0,
    shots_total     INTEGER DEFAULT 0,
    shots_on_target INTEGER DEFAULT 0,
    passes_total    INTEGER DEFAULT 0,
    pass_accuracy   NUMERIC(5, 2),
    tackles         INTEGER DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (player_id, match_id)
);

CREATE INDEX IF NOT EXISTS idx_player_stats_player ON player_stats(player_id);
CREATE INDEX IF NOT EXISTS idx_player_stats_match  ON player_stats(match_id);

-- ============================================================
-- Bookmaker odds (from Football-Data.co.uk CSVs)
-- ============================================================
CREATE TABLE IF NOT EXISTS odds (
    id              SERIAL PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    bookmaker       TEXT NOT NULL DEFAULT 'B365',  -- Bet365 is standard in FD.co.uk CSVs
    home_odds       NUMERIC(8, 4),
    draw_odds       NUMERIC(8, 4),
    away_odds       NUMERIC(8, 4),
    -- Implied probabilities (normalised to sum to 1, removing overround)
    implied_home    NUMERIC(8, 6),
    implied_draw    NUMERIC(8, 6),
    implied_away    NUMERIC(8, 6),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_id);

-- ============================================================
-- Model predictions log
-- ============================================================
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   SERIAL PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id) ON DELETE CASCADE,
    tier            INTEGER NOT NULL CHECK (tier IN (1, 2)),  -- 1=pre-lineup, 2=post-lineup
    model_version   TEXT NOT NULL,
    prob_home       NUMERIC(8, 6) NOT NULL,
    prob_draw       NUMERIC(8, 6) NOT NULL,
    prob_away       NUMERIC(8, 6) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, tier, model_version)
);

CREATE INDEX IF NOT EXISTS idx_predictions_match ON predictions(match_id);
CREATE INDEX IF NOT EXISTS idx_predictions_tier  ON predictions(tier);
