"""
Database loader — all queries against the 'diet' DB.
No hardcoded nutrient IDs or DRI values; everything comes from the DB.
"""
import mysql.connector
import pandas as pd
from config import DB_CONFIG, NUTRIENT_IDS


def _connect():
    return mysql.connector.connect(**DB_CONFIG)


def _cursor_to_df(cursor, columns):
    rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def get_foods(user_id: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      food_id, name, foodGroupId, cost, co2, preparingTime, cookingTime,
      preference  (from user_foods; falls back to foods.preference if no entry)

    Uses LEFT JOIN so all 405 foods are always returned.
    Preference fallback: COALESCE(uf.preference, f.preference, 0).
    """
    sql = """
        SELECT
            f.id            AS food_id,
            f.name          AS name,
            f.foodGroupId   AS foodGroupId,
            f.cost          AS cost,
            f.co2           AS co2,
            f.preparingTime AS preparingTime,
            f.cookingTime   AS cookingTime,
            COALESCE(uf.preference, f.preference, 0) AS preference
        FROM foods f
        LEFT JOIN user_foods uf
               ON uf.foodId = f.id AND uf.userId = %s
        ORDER BY f.id
    """
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, (user_id,))
        cols = [d[0] for d in cursor.description]
        df = _cursor_to_df(cursor, cols)
    finally:
        conn.close()
    return df


def get_nutrients_for_foods(food_ids: list) -> pd.DataFrame:
    """
    Returns pivoted DataFrame: food_id (index) x nutrient_id (columns).
    Only the 5 target nutrient IDs from config.NUTRIENT_IDS.
    Missing values filled with 0.
    """
    if not food_ids:
        return pd.DataFrame(index=[], columns=NUTRIENT_IDS).fillna(0.0)

    placeholders = ",".join(["%s"] * len(food_ids))
    nutrient_placeholders = ",".join(["%s"] * len(NUTRIENT_IDS))

    sql = f"""
        SELECT fn.foodId AS food_id, fn.nutrientId AS nutrient_id, fn.quantity
        FROM food_nutrients fn
        WHERE fn.foodId IN ({placeholders})
          AND fn.nutrientId IN ({nutrient_placeholders})
    """
    params = list(food_ids) + list(NUTRIENT_IDS)

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(sql, params)
        cols = [d[0] for d in cursor.description]
        df = _cursor_to_df(cursor, cols)
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(0.0, index=food_ids, columns=NUTRIENT_IDS)

    pivot = df.pivot_table(
        index="food_id", columns="nutrient_id", values="quantity", fill_value=0.0
    )
    # Ensure all 5 nutrient columns exist
    for nid in NUTRIENT_IDS:
        if nid not in pivot.columns:
            pivot[nid] = 0.0

    pivot = pivot[NUTRIENT_IDS]
    pivot = pivot.reindex(food_ids, fill_value=0.0)
    return pivot


def get_dri(user_id: int) -> dict:
    """
    Returns dict {nutrient_id: (RLL, RUL)} for the 5 target nutrients.
    Joins dri table with user table on age range and gender.
    """
    nutrient_placeholders = ",".join(["%s"] * len(NUTRIENT_IDS))
    sql = f"""
        SELECT d.nutrient_id, d.RLL, d.RUL
        FROM dri d
        JOIN user u ON u.age BETWEEN d.low_age AND d.up_age
                   AND LOWER(u.gender) = LOWER(d.gender)
        WHERE u.id = %s
          AND d.nutrient_id IN ({nutrient_placeholders})
    """
    params = [user_id] + list(NUTRIENT_IDS)

    conn = _connect()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return {row["nutrient_id"]: (float(row["RLL"]), float(row["RUL"])) for row in rows}


def get_food_groups(food_ids: list) -> dict:
    """Returns dict {food_id: foodGroupId}."""
    if not food_ids:
        return {}

    placeholders = ",".join(["%s"] * len(food_ids))
    sql = f"SELECT id AS food_id, foodGroupId FROM foods WHERE id IN ({placeholders})"

    conn = _connect()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(sql, food_ids)
        rows = cursor.fetchall()
    finally:
        conn.close()

    return {row["food_id"]: row["foodGroupId"] for row in rows}


def load_all_for_user(user_id: int) -> dict:
    """
    Convenience loader — returns everything needed for one user.
    food_preferences: dict {food_id: preference_value} for the Chromosome decoder.
    """
    foods_df = get_foods(user_id)
    food_ids = foods_df["food_id"].tolist()

    nutrient_matrix = get_nutrients_for_foods(food_ids)
    dri = get_dri(user_id)
    food_groups = get_food_groups(food_ids)
    food_preferences = dict(zip(foods_df["food_id"], foods_df["preference"]))

    return {
        "foods_df": foods_df,
        "nutrient_matrix": nutrient_matrix,
        "dri": dri,
        "food_groups": food_groups,
        "food_ids": food_ids,
        "food_preferences": food_preferences,
    }


if __name__ == "__main__":
    for uid in [1, 2]:
        print(f"\n=== Testing db_loader for User {uid} ===")
        data = load_all_for_user(uid)
        print(f"  Foods loaded:    {len(data['food_ids'])}")
        print(f"  Nutrient matrix: {data['nutrient_matrix'].shape}")
        print(f"  DRI entries:     {len(data['dri'])}")
        print(f"  Food groups:     {len(data['food_groups'])}")
        print(f"  DRI values:")
        for nid, (rll, rul) in sorted(data["dri"].items()):
            print(f"    nutrient_id={nid}: RLL={rll:.1f}, RUL={rul:.1f}")
        print(f"  Sample foods (first 3):")
        print(data["foods_df"][["food_id", "name", "cost", "co2", "preference"]].head(3).to_string(index=False))
        grp_counts = data["foods_df"]["foodGroupId"].nunique()
        print(f"  Distinct food groups: {grp_counts}")
