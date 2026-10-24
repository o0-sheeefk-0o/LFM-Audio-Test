"""システムプロンプト長の検証用に、指定トークン数ぶんのダミー資料（社員名簿）を作る

各行に固有の情報（社員番号・所属・内線・好物）が入っているので、
「社員番号1234の人の内線番号は？」のように聞けば、長いプロンプトの奥の情報を参照できているかも確認できる。
"""

import random
from collections.abc import Callable

HEADER = "\n\n# 参考資料：社員名簿\n質問されたら、以下の名簿を参照して答えてください。\n"

SURNAMES = ["佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤", "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "斎藤", "清水", "森"]
GIVEN_NAMES = ["太郎", "花子", "健太", "美咲", "翔", "陽菜", "大輔", "さくら", "拓海", "結衣", "蓮", "葵", "悠斗", "凛", "颯太"]
DEPARTMENTS = ["営業部", "経理部", "開発部", "人事部", "総務部", "企画部", "広報部", "品質管理部", "法務部", "購買部"]
FOODS = ["ラーメン", "寿司", "カレー", "うどん", "焼肉", "パスタ", "天ぷら", "そば", "餃子", "お好み焼き", "牛丼", "ハンバーグ"]


def make_filler(target_tokens: int, count_tokens: Callable[[str], int], seed: int = 0) -> str:
    rng = random.Random(seed)
    lines = [HEADER]
    total = count_tokens(HEADER)
    employee_id = 1000
    while total < target_tokens:
        line = (
            f"- 社員番号{employee_id}：{rng.choice(SURNAMES)}{rng.choice(GIVEN_NAMES)}（{rng.choice(DEPARTMENTS)}）、"
            f"内線{rng.randint(1000, 9999)}、好物は{rng.choice(FOODS)}。\n"
        )
        lines.append(line)
        total += count_tokens(line)
        employee_id += 1
    return "".join(lines)
