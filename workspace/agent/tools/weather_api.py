import os
import requests

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(city: str):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={city}&appid={OPENWEATHER_API_KEY}&units=metric&lang=kr"
    )

    response = requests.get(url)
    data = response.json()

    if response.status_code != 200:
        return {"error": data.get("message", "API 요청 실패")}

    return {
        "도시": data["name"],
        "기온": data["main"]["temp"],
        "체감기온": data["main"]["feels_like"],
        "날씨": data["weather"][0]["description"],
        "습도": data["main"]["humidity"],
        "풍속": data["wind"]["speed"],
    }
