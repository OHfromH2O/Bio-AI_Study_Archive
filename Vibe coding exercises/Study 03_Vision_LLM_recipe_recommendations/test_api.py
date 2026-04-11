import os
import io
import base64
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}


def test_text(model: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def test_image(model: str, image_b64: str, prompt: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    resp = requests.post(BASE_URL, headers=HEADERS, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def make_test_image() -> Image.Image:
    """사과 모양의 테스트 이미지 생성"""
    img = Image.new("RGB", (300, 300), color="white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([60, 80, 240, 260], fill="red", outline="darkred", width=3)
    draw.rectangle([138, 40, 162, 85], fill="saddlebrown")
    draw.ellipse([110, 30, 170, 80], fill="green", outline="darkgreen", width=2)
    draw.text((90, 280), "Test Image: Apple", fill="black")
    return img


if __name__ == "__main__":
    # ── 1. 텍스트 인식 테스트 ──────────────────────────────────────────
    TEXT_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
    print(f"[텍스트] 모델: {TEXT_MODEL}")
    try:
        answer = test_text(TEXT_MODEL, "대한민국의 수도는 어디인가요? 한 문장으로 답해주세요.")
        print(f"  응답: {answer}\n")
    except Exception as e:
        print(f"  오류: {e}\n")

    # ── 2. 이미지 인식 테스트 ──────────────────────────────────────────
    IMAGE_MODEL = "nvidia/nemotron-nano-12b-v2-vl:free"
    print(f"[이미지] 모델: {IMAGE_MODEL}")
    try:
        img = make_test_image()
        b64 = image_to_base64(img)
        answer = test_image(IMAGE_MODEL, b64, "이 이미지에 무엇이 있나요? 한 문장으로 답해주세요.")
        print(f"  응답: {answer}\n")
    except Exception as e:
        print(f"  오류: {e}\n")
