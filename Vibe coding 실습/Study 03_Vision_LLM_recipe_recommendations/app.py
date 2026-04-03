import os
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder=".")
API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)


@app.route("/api/recognize", methods=["POST"])
def recognize():
    """이미지를 받아 Vision 모델로 재료를 인식한다."""
    data = request.get_json()
    image_b64 = data.get("image")  # data URI 형식
    if not image_b64:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    payload = {
        "model": "nvidia/nemotron-nano-12b-v2-vl:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {
                        "type": "text",
                        "text": (
                            "이 냉장고 사진에서 보이는 식재료를 모두 파악하여, "
                            "JSON 배열 형식으로만 반환해 주세요.\n"
                            '형식 예시: ["달걀", "우유", "당근", "버터"]\n'
                            "식재료가 아닌 설명은 포함하지 마세요."
                        ),
                    },
                ],
            }
        ],
    }

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        ingredients = parse_ingredients(content)
        return jsonify({"ingredients": ingredients})
    except requests.Timeout:
        return jsonify({"error": "API 응답 시간이 초과되었습니다. 다시 시도해 주세요."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recipe", methods=["POST"])
def recipe():
    """재료 목록을 받아 Text LLM으로 레시피 3개를 생성한다."""
    data = request.get_json()
    ingredients = data.get("ingredients", [])
    extra_conditions = data.get("conditions", "")  # Step 3 프로필 조건 (선택)

    if len(ingredients) < 2:
        return jsonify({"error": "재료를 2개 이상 입력해 주세요."}), 400

    ingredient_str = ", ".join(ingredients)
    condition_line = f"\n단, 다음 조건을 반드시 지켜주세요: {extra_conditions}" if extra_conditions else ""

    prompt = (
        f"다음 재료들로 만들 수 있는 요리 레시피 3가지를 JSON 배열로 반환해 주세요.\n"
        f"재료 목록: {ingredient_str}{condition_line}\n"
        "각 레시피의 형식:\n"
        "[\n"
        "  {\n"
        '    "name": "요리 이름",\n'
        '    "description": "한 줄 설명",\n'
        '    "cooking_time": "소요 시간 (예: 20분)",\n'
        '    "difficulty": "난이도 (쉬움/보통/어려움)",\n'
        '    "ingredients": ["재료1 적정량", "재료2 적정량"],\n'
        '    "steps": ["1단계 설명", "2단계 설명"]\n'
        "  }\n"
        "]\n"
        "JSON 배열 외의 텍스트는 포함하지 마세요."
    )

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "nvidia/nemotron-3-super-120b-a12b:free",
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        recipes = parse_recipes(content)
        return jsonify({"recipes": recipes})
    except requests.Timeout:
        return jsonify({"error": "API 응답 시간이 초과되었습니다. 다시 시도해 주세요."}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def parse_recipes(text: str) -> list:
    """JSON 배열 파싱 — 실패 시 폴백으로 빈 구조 반환."""
    import json
    import re

    # 1. 코드 펜스 제거
    text = re.sub(r"```(?:json)?", "", text).strip()

    # 2. JSON 배열 직접 파싱
    try:
        start, end = text.index("["), text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # 3. 폴백: 개별 JSON 오브젝트 추출
    recipes = []
    for match in re.finditer(r"\{[^{}]+\}", text, re.DOTALL):
        try:
            recipes.append(json.loads(match.group()))
        except json.JSONDecodeError:
            pass
    return recipes


def parse_ingredients(text: str) -> list[str]:
    """JSON 배열 파싱 — 실패 시 텍스트에서 재료 추출(폴백)."""
    import json
    import re

    # 1. JSON 배열 직접 파싱
    try:
        start, end = text.index("["), text.rindex("]") + 1
        return json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # 2. 폴백: 줄 단위 또는 쉼표 구분으로 재료 추출
    cleaned = re.sub(r"[\"'\[\]{}]", "", text)
    items = [item.strip() for item in re.split(r"[,\n•·-]", cleaned) if item.strip()]
    return [item for item in items if 1 <= len(item) <= 20]


if __name__ == "__main__":
    app.run(debug=True, port=5000)
