"""サンプルツールの定義・モック実行・ツールコールのパース

LFM2 系のツールコール書式:
  - 定義:   system ターンに `List of tools: <|tool_list_start|>[{...}, ...]<|tool_list_end|>`
  - 呼び出し: assistant が `<|tool_call_start|>[func(arg="value")]<|tool_call_end|>` を出力（Pythonic）
  - 結果:   `<|im_start|>tool\n<|tool_response_start|>...<|tool_response_end|><|im_end|>`
"""

import ast
import json
import operator
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

TOOL_CALL_RE = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)

SAMPLE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "指定した都市の現在の天気と気温を取得する",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "都市名（例: 東京）"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "get_current_time",
        "description": "現在の日付と時刻を取得する",
        "parameters": {
            "type": "object",
            "properties": {
                "timezone": {"type": "string", "description": "IANA タイムゾーン名（例: Asia/Tokyo）。省略時は日本時間"},
            },
            "required": [],
        },
    },
    {
        "name": "calculate",
        "description": "四則演算などの数式を計算して結果を返す",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "計算式（例: (1200 + 800) * 1.1）"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_restaurants",
        "description": "エリアと料理ジャンルからレストランを検索する",
        "parameters": {
            "type": "object",
            "properties": {
                "area": {"type": "string", "description": "エリア名（例: 渋谷）"},
                "cuisine": {"type": "string", "description": "料理ジャンル（例: ラーメン）"},
            },
            "required": ["area"],
        },
    },
    {
        "name": "set_timer",
        "description": "指定した分数のタイマーをセットする",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "タイマーの長さ（分）"},
                "label": {"type": "string", "description": "タイマーの名前"},
            },
            "required": ["minutes"],
        },
    },
]

SAMPLE_TOOLS_JSON = json.dumps(SAMPLE_TOOLS, ensure_ascii=False, indent=2)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.arguments.items())
        return f"{self.name}({args})"


def load_tools(tools_json: str) -> list[dict[str, Any]]:
    tools = json.loads(tools_json)
    if isinstance(tools, dict):
        tools = [tools]
    if not isinstance(tools, list):
        raise ValueError("ツール定義は JSON の配列で指定してください")
    return tools


def parse_tool_calls(body: str) -> list[ToolCall]:
    """`<|tool_call_start|>` と `<|tool_call_end|>` に挟まれた中身をパースする

    Pythonic 形式 `[f(a=1), g(b="x")]` と JSON 形式 `[{"name": ..., "arguments": {...}}]` の両方を受け付ける。
    """
    body = body.strip()
    try:
        return _parse_pythonic(body)
    except (SyntaxError, ValueError):
        pass

    data = json.loads(body)
    if isinstance(data, dict):
        data = [data]
    calls = []
    for item in data:
        args = item.get("arguments", item.get("parameters", {}))
        if isinstance(args, str):
            args = json.loads(args)
        calls.append(ToolCall(name=item["name"], arguments=args))
    return calls


def _parse_pythonic(body: str) -> list[ToolCall]:
    node = ast.parse(body, mode="eval").body
    nodes = node.elts if isinstance(node, ast.List) else [node]
    calls = []
    for call in nodes:
        if not isinstance(call, ast.Call):
            raise ValueError(f"関数呼び出しではありません: {ast.unparse(call)}")
        args = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg}
        for i, arg in enumerate(call.args):
            args[f"arg{i}"] = ast.literal_eval(arg)
        calls.append(ToolCall(name=ast.unparse(call.func), arguments=args))
    return calls


def extract_tool_calls(text: str) -> tuple[list[ToolCall], list[str]]:
    """生成テキストからツールコールを抜き出す。戻り値は (パース成功したコール, パース失敗した生文字列)"""
    calls: list[ToolCall] = []
    errors: list[str] = []
    for body in TOOL_CALL_RE.findall(text):
        try:
            calls.extend(parse_tool_calls(body))
        except Exception:
            errors.append(body)
    return calls, errors


# ---------------------------------------------------------------------------
# モック実装
# ---------------------------------------------------------------------------


def _rng(*keys: Any) -> random.Random:
    return random.Random("|".join(map(str, keys)))


def get_weather(city: str = "東京", **_: Any) -> dict[str, Any]:
    rng = _rng(city, datetime.now().date())
    return {
        "city": city,
        "weather": rng.choice(["晴れ", "くもり", "雨", "晴れ時々くもり", "くもり時々雨"]),
        "temperature_c": rng.randint(5, 32),
        "humidity_percent": rng.randint(30, 90),
    }


def get_current_time(timezone: str = "Asia/Tokyo", **_: Any) -> dict[str, Any]:
    now = datetime.now(ZoneInfo(timezone))
    weekday = "月火水木金土日"[now.weekday()]
    return {"timezone": timezone, "datetime": now.strftime(f"%Y年%m月%d日（{weekday}）%H時%M分")}


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_arith(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_arith(node.left), _eval_arith(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_arith(node.operand))
    raise ValueError(f"サポートしていない式です: {ast.unparse(node)}")


def calculate(expression: str, **_: Any) -> dict[str, Any]:
    normalized = expression.translate(str.maketrans("０１２３４５６７８９＋－×÷（）．", "0123456789+-*/().")).replace("^", "**")
    result = _eval_arith(ast.parse(normalized, mode="eval").body)
    if isinstance(result, float):
        result = round(result, 10)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return {"expression": expression, "result": result}


def search_restaurants(area: str = "", cuisine: str = "", **_: Any) -> dict[str, Any]:
    rng = _rng(area, cuisine)
    genre = cuisine or rng.choice(["和食", "イタリアン", "中華", "焼肉"])
    names = ["一番亭", "さくら", "まんぷく", "ことぶき", "はなまる", "つきみ", "あおぞら"]
    results = [
        {
            "name": f"{genre}処 {name}",
            "rating": round(rng.uniform(3.2, 4.8), 1),
            "walk_minutes_from_station": rng.randint(1, 12),
        }
        for name in rng.sample(names, 3)
    ]
    return {"area": area, "cuisine": genre, "results": results}


def set_timer(minutes: int = 1, label: str = "タイマー", **_: Any) -> dict[str, Any]:
    ends_at = datetime.now(ZoneInfo("Asia/Tokyo")) + timedelta(minutes=int(minutes))
    return {"status": "started", "label": label, "minutes": int(minutes), "ends_at": ends_at.strftime("%H時%M分")}


TOOL_IMPLS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_restaurants": search_restaurants,
    "set_timer": set_timer,
}


def execute_tool(call: ToolCall, available: list[dict[str, Any]]) -> dict[str, Any]:
    """ツールを実行する。定義にはあるが実装がないツール（UI で追加したもの）はモック成功を返す"""
    if call.name not in {t.get("name") for t in available}:
        return {"error": f"unknown tool: {call.name}"}
    impl = TOOL_IMPLS.get(call.name)
    if impl is None:
        return {"status": "ok", "note": "mock response", "tool": call.name, "arguments": call.arguments}
    try:
        return impl(**call.arguments)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
