"""Cliente IMAP leve para leitura e busca de e-mails por voz (Gmail, Outlook, Yahoo ou customizado)."""

from __future__ import annotations

import email
import imaplib
import logging
import os
import re
from email.header import decode_header
from email.message import Message
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_IMAP_HOSTS: Dict[str, str] = {
    "gmail.com": "imap.gmail.com",
    "googlemail.com": "imap.gmail.com",
    "outlook.com": "outlook.office365.com",
    "hotmail.com": "outlook.office365.com",
    "live.com": "outlook.office365.com",
    "msn.com": "outlook.office365.com",
    "yahoo.com": "imap.mail.yahoo.com",
    "yahoo.com.br": "imap.mail.yahoo.com",
    "icloud.com": "imap.mail.me.com",
}


def _resolve_credentials() -> Tuple[str, str, str]:
    addr = os.getenv("EMAIL_ADDRESS", "").strip()
    pwd = os.getenv("EMAIL_APP_PASSWORD", "").strip().replace(" ", "")
    host = os.getenv("EMAIL_IMAP_HOST", "").strip()
    if not host and "@" in addr:
        domain = addr.split("@", 1)[1].lower()
        host = _IMAP_HOSTS.get(domain, f"imap.{domain}")
    return addr, pwd, host


def is_email_configured() -> bool:
    addr, pwd, host = _resolve_credentials()
    return bool(addr and pwd and host)


def _decode_mime_words(raw: str | None) -> str:
    if not raw:
        return ""
    parts: List[str] = []
    for content, charset in decode_header(raw):
        if isinstance(content, bytes):
            enc = charset or "utf-8"
            try:
                parts.append(content.decode(enc, errors="replace"))
            except LookupError:
                parts.append(content.decode("utf-8", errors="replace"))
        else:
            parts.append(str(content))
    return " ".join(parts).strip()


def _extract_plain_body(msg: Message, max_chars: int = 320) -> str:
    body_text = ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp.lower():
                continue
            if ctype == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    body_text = payload.decode(charset, errors="replace")
                    break
            elif ctype == "text/html" and not body_text:
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    html = payload.decode(charset, errors="replace")
                    body_text = re.sub(r"<[^>]+>", " ", html)
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            raw_text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                raw_text = re.sub(r"<[^>]+>", " ", raw_text)
            body_text = raw_text

    cleaned = re.sub(r"\s+", " ", body_text).strip()
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 1].rstrip() + "…"
    return cleaned


def _fetch_messages(criteria: str, limit: int = 5) -> str:
    addr, pwd, host = _resolve_credentials()
    if not addr or not pwd:
        return (
            "E-mail ainda não configurado no arquivo .env. "
            "Preencha EMAIL_ADDRESS e EMAIL_APP_PASSWORD (Senha de App do Gmail/Outlook)."
        )

    mail: imaplib.IMAP4_SSL | None = None
    try:
        mail = imaplib.IMAP4_SSL(host, timeout=6.0)
        mail.login(addr, pwd)
        mail.select("INBOX", readonly=True)

        status, data = mail.search(None, criteria)
        if status != "OK" or not data or not data[0]:
            return "Nenhum e-mail encontrado para esse filtro."

        ids = data[0].split()
        total_found = len(ids)
        recent_ids = list(reversed(ids[-max(1, min(limit, 10)) :]))

        summaries: List[str] = []
        for idx, msg_id in enumerate(recent_ids, start=1):
            res, msg_data = mail.fetch(msg_id, "(RFC822)")
            if res != "OK" or not msg_data:
                continue
            raw_email = msg_data[0][1]
            if not isinstance(raw_email, bytes):
                continue
            msg = email.message_from_bytes(raw_email)
            subject = _decode_mime_words(msg.get("Subject")) or "(Sem assunto)"
            sender = _decode_mime_words(msg.get("From"))
            snippet = _extract_plain_body(msg, max_chars=260)
            summaries.append(
                f"{idx}) De: {sender} | Assunto: {subject} | Trecho: {snippet}"
            )

        header = f"Total encontrado: {total_found} e-mail(s). Mostrando os {len(summaries)} mais recentes:"
        return header + "\n" + "\n".join(summaries)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Erro ao consultar e-mails via IMAP: %s", exc)
        return f"Não foi possível acessar a caixa de e-mail ({exc}). Verifique EMAIL_ADDRESS e EMAIL_APP_PASSWORD no .env."
    finally:
        if mail is not None:
            try:
                mail.logout()
            except Exception:  # noqa: BLE001
                pass


def read_recent_emails(unread_only: bool = True, limit: int = 4) -> str:
    """Lê os últimos e-mails não lidos (ou recentes) da caixa de entrada."""
    criteria = "UNSEEN" if unread_only else "ALL"
    return _fetch_messages(criteria, limit=limit)


def search_emails(query: str, limit: int = 4) -> str:
    """Pesquisa e-mails na caixa de entrada por termo no assunto, remetente ou texto."""
    cleaned = query.strip()
    if not cleaned:
        return read_recent_emails(unread_only=False, limit=limit)
    # Remove acentos para compatibilidade com servidores IMAP ASCII
    ascii_term = cleaned.encode("ascii", errors="ignore").decode("ascii").strip()
    term = ascii_term or cleaned
    criteria = f'(OR SUBJECT "{term}" FROM "{term}")'
    return _fetch_messages(criteria, limit=limit)

