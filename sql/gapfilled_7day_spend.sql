-- Materialize user×day features where missing days count as 0
WITH bounds AS (
    SELECT user_id,
        MIN(CAST(event_timestamp AS DATE)) AS start_dt,
        MAX(CAST(event_timestamp AS DATE)) AS end_dt
    FROM icecat.{namespace}.raw_events
    GROUP BY 1
),
calendar AS (
    SELECT b.user_id,
        d::DATE AS dt
    FROM bounds b,
        LATERAL (
            SELECT *
            FROM generate_series(b.start_dt, b.end_dt, INTERVAL 1 DAY)
        ) gs(d)
),
daily AS (
    SELECT user_id,
        CAST(event_timestamp AS DATE) AS dt,
        SUM(amount) AS daily_spend
    FROM icecat.{namespace}.raw_events
    GROUP BY 1,
        2
),
daily_filled AS (
    SELECT c.user_id,
        c.dt,
        COALESCE(d.daily_spend, 0) AS daily_spend
    FROM calendar c
        LEFT JOIN daily d ON d.user_id = c.user_id
        AND d.dt = c.dt
)
SELECT user_id,
    dt,
    CAST(
        SUM(daily_spend) OVER (
            PARTITION BY user_id
            ORDER BY dt ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) / 7.0 AS DECIMAL(15, 2)
    ) AS spending_mean_7d
FROM daily_filled;