"""Coletor paralelo em tempo real para o 'Modo Briefing' ('Ren, me atualiza' / 'Bom dia, Ren').

Busca simultaneamente (em < 1.2s):
1. Cidade aproximada + Clima atual e previsão de hoje (Open-Meteo / ip-api).
2. Cotações em tempo real de Dólar (USD/BRL), Euro (EUR/BRL) e Bitcoin (BTC).
3. As 3 principais manchetes do momento no Brasil (Google News RSS PT-BR).
4. Memórias/lembretes salvos do usuário e status rápido do sistema.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Sequence

from src.system.email_client import is_email_configured, read_recent_emails
from src.system.pc_actions import get_pc_status

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

_WMO_WEATHER_PT: Dict[int, str] = {
    0: "céu limpo",
    1: "predominantemente limpo",
    2: "parcialmente nublado",
    3: "encoberto",
    45: "neblina",
    48: "nevoeiro",
    51: "garoa leve",
    53: "garoa moderada",
    55: "garoa intensa",
    61: "chuva fraca",
    63: "chuva moderada",
    65: "chuva forte",
    80: "pancadas de chuva leves",
    81: "pancadas de chuva moderadas",
    82: "pancadas de chuva fortes",
    95: "trovoadas",
    96: "trovoadas com granizo",
}


def _http_get_json(url: str, timeout: float = 3.5) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def fetch_weather_summary(city_override: str = "") -> str:
    """Obtém clima atual e máxima/mínima do dia para a cidade informada, USER_LOCATION ou via IP."""
    try:
        raw_city = (
            city_override.strip()
            or os.getenv("USER_LOCATION", "Santa Mariana, Paraná, Brasil").strip()
        )
        parts = [p.strip() for p in raw_city.split(",") if p.strip()]
        city_name = parts[0] if parts else ""
        state_hint = parts[1].lower() if len(parts) > 1 else ""

        city = raw_city
        lat: float | None = None
        lon: float | None = None

        if city_name:
            geo_url = (
                "https://geocoding-api.open-meteo.com/v1/search?"
                f"name={urllib.parse.quote_plus(city_name)}&count=5&language=pt&format=json"
            )
            geo = _http_get_json(geo_url)
            results = geo.get("results") or []
            if results:
                chosen = results[0]
                if state_hint:
                    for candidate in results:
                        admin1 = str(candidate.get("admin1", "")).lower()
                        country = str(candidate.get("country_code", "")).lower()
                        if state_hint in admin1 or (state_hint in {"pr", "parana", "paraná"} and "paran" in admin1):
                            chosen = candidate
                            break
                        if country == "br" and chosen.get("country_code", "").lower() != "br":
                            chosen = candidate
                lat = float(chosen["latitude"])
                lon = float(chosen["longitude"])
                admin1_label = chosen.get("admin1", "")
                city = (
                    f"{chosen.get('name', city_name)}, {admin1_label}".strip(", ")
                    if admin1_label
                    else str(chosen.get("name", city_name))
                )

        if lat is None or lon is None:
            ip_info = _http_get_json("http://ip-api.com/json/?fields=status,city,regionName,lat,lon")
            if ip_info.get("status") == "success":
                lat = float(ip_info["lat"])
                lon = float(ip_info["lon"])
                city = f"{ip_info.get('city', '')}, {ip_info.get('regionName', '')}".strip(", ")

        if lat is None or lon is None:
            return "Clima indisponível no momento."

        meteo_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,apparent_temperature,weather_code"
            "&daily=temperature_2m_max,temperature_2m_min,precipitation_probability_max"
            "&timezone=auto&forecast_days=1"
        )
        data = _http_get_json(meteo_url)
        cur = data.get("current", {})
        daily = data.get("daily", {})

        temp = cur.get("temperature_2m")
        feels = cur.get("apparent_temperature")
        code = int(cur.get("weather_code", 0))
        desc = _WMO_WEATHER_PT.get(code, "tempo variável")

        t_max = (daily.get("temperature_2m_max") or [None])[0]
        t_min = (daily.get("temperature_2m_min") or [None])[0]
        rain_prob = (daily.get("precipitation_probability_max") or [0])[0]

        return (
            f"Clima em {city}: {temp:.0f}°C agora (sensação de {feels:.0f}°C), {desc}. "
            f"Hoje a máxima é de {t_max:.0f}°C e mínima de {t_min:.0f}°C, "
            f"com {rain_prob}% de chance de chuva."
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Falha ao buscar clima: %s", exc)
        return "Clima: consulte via Google Search se necessário."


def fetch_market_quotes() -> str:
    """Busca cotação atual do Dólar (USD/BRL), Euro (EUR/BRL) e Bitcoin (BTC/BRL)."""
    try:
        data = _http_get_json(
            "https://economia.awesomeapi.com.br/json/last/USD-BRL,EUR-BRL,BTC-BRL"
        )
        usd = data.get("USDBRL", {})
        eur = data.get("EURBRL", {})
        btc = data.get("BTCBRL", {})

        usd_bid = float(usd.get("bid", 0.0))
        usd_pct = float(usd.get("pctChange", 0.0))
        eur_bid = float(eur.get("bid", 0.0))
        btc_bid = float(btc.get("bid", 0.0))

        return (
            f"Cotações agora: Dólar a R$ {usd_bid:.2f} ({usd_pct:+.2f}% no dia), "
            f"Euro a R$ {eur_bid:.2f}, Bitcoin a R$ {btc_bid:,.0f}."
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Falha ao buscar cotações: %s", exc)
        return "Cotações: consulte via Google Search se necessário."


def fetch_top_brazil_news(limit: int = 3) -> str:
    """Busca as principais manchetes em tempo real no Google News Brasil (RSS)."""
    url = "https://news.google.com/rss?hl=pt-BR&gl=BR&ceid=BR:pt-419"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(req, timeout=3.5) as resp:  # noqa: S310
            xml_bytes = resp.read()
        root = ET.fromstring(xml_bytes)  # noqa: S314
        headlines: List[str] = []
        for item in root.findall("./channel/item"):
            title_el = item.find("title")
            if title_el is not None and title_el.text:
                headlines.append(title_el.text.strip())
            if len(headlines) >= limit:
                break
        if not headlines:
            return "Manchetes: nenhuma encontrada no feed."
        joined = " | ".join(f"{i+1}) {h}" for i, h in enumerate(headlines))
        return f"Principais notícias do momento no Brasil: {joined}"
    except Exception as exc:  # noqa: BLE001
        logger.debug("Falha ao buscar notícias: %s", exc)
        return "Notícias: consulte via Google Search se necessário."


def build_executive_briefing(
    city: str = "", memories: Sequence[str] = ()
) -> str:
    """Executa todas as coletas do Briefing em paralelo e retorna um dossiê completo."""
    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_weather = pool.submit(fetch_weather_summary, city)
        fut_markets = pool.submit(fetch_market_quotes)
        fut_news = pool.submit(fetch_top_brazil_news, 3)
        fut_email = (
            pool.submit(read_recent_emails, True, 3)
            if is_email_configured()
            else None
        )

        weather_str = fut_weather.result()
        markets_str = fut_markets.result()
        news_str = fut_news.result()
        email_str = fut_email.result() if fut_email is not None else ""

    now_str = datetime.now().strftime("%H:%M")
    parts = [
        f"Horário atual: {now_str}.",
        weather_str,
        markets_str,
        news_str,
        f"Sistema: {get_pc_status()}",
    ]
    if email_str:
        parts.append(f"E-mails não lidos: {email_str}")
    if memories:
        parts.append("Memórias/Lembretes do usuário: " + "; ".join(memories[-8:]))
    return "\n".join(parts)

