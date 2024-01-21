import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=str, default="./docs/fraud.csv", help="Raw file path"
    )
    parser.add_argument(
        "--ip_file",
        type=str,
        default="./docs/ip_country.xlsx",
        help="IP-Country mapping file",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="./data/train.csv",
        help="Save path for processed csv(train data)",
    )
    parser.add_argument("--save_json", action="store_true", help="Save map")
    parser.add_argument("--load_json", action="store_true", help="Use saved map")
    args = parser.parse_args()
    return args


def get_map(category, df):
    dic = {}
    temp = df[category].drop_duplicates()
    temp_array = temp.values
    for i, x in enumerate(temp_array):
        dic[x] = i
    return dic


def to_datetime(x):
    try:
        time_format = "%Y-%m-%d %H:%M"  # For no-label file
        time = datetime.strptime(x, time_format)
    except:
        time_format = "%m/%d/%Y %H:%M"  # For train csv file
        time = datetime.strptime(x, time_format)
    return time


def to_minuets(x):
    return x.total_seconds() / 60


def find_country(number, df):
    country = df[
        (df["lower_bound_ip_address"] <= number)
        & (df["upper_bound_ip_address"] >= number)
    ]["country"]
    return country.iloc[0] if not country.empty else "Unknown"


def drop_unnecessary(df: pd.DataFrame):
    to_drop_column = [
        "user_id",
        "signup_time",
        "purchase_time",
        "device_id",
        "ip_address",
    ]
    df = df.drop(to_drop_column, axis=1)
    return df


def pre_process(df: pd.DataFrame, ip):
    df["signup_time"] = df["signup_time"].apply(to_datetime)
    df["purchase_time"] = df["purchase_time"].apply(to_datetime)

    df.insert(loc=3, column="time_delta", value=None)
    df.insert(loc=11, column="ip_country", value=None)

    df["time_delta"] = df["purchase_time"] - df["signup_time"]
    df["time_delta"] = df["time_delta"].apply(to_minuets)
    df["ip_country"] = df["ip_address"].apply(lambda x: find_country(x, ip))
    df = drop_unnecessary(df)
    return df


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)
    return dictionary


def main(args):
    # DataLoad
    csv_path = Path(args.csv)
    ip_path = Path(args.ip_file)

    df = pd.read_csv(csv_path, index_col=0)
    ip = pd.read_excel(ip_path)
    df = pre_process(df, ip)

    if args.load_json:
        source_map = load_json("./data/source_map.json")
        browser_map = load_json("./data/browser_map.json")
        sex_map = load_json("./data/sex_map.json")
        country_map = load_json("./data/ip_country_map.json")
    else:
        source_map = get_map("source", df=df)
        browser_map = get_map("browser", df=df)
        sex_map = get_map("sex", df=df)
        country_map = get_map("ip_country", df=df)

    df["source"] = df["source"].map(source_map)
    df["browser"] = df["browser"].map(browser_map)
    df["sex"] = df["sex"].map(sex_map)
    df["ip_country"] = df["ip_country"].map(country_map)
    df.to_csv(args.save_name, encoding="utf-8", index=False)

    if args.save_json:
        save_json("./data/source_map.json", source_map)
        save_json("./data/browser_map.json", browser_map)
        save_json("./data/sex_map.json", sex_map)
        save_json("./data/ip_country_map.json", country_map)


if __name__ == "__main__":
    args = get_args()
    main(args)
