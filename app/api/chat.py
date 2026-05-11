from __future__ import annotations

import os
import re

from flask import Blueprint, current_app, jsonify, request

from ..services.conversation_store import InMemoryConversationStore
from ..services.knowledge_base import KnowledgeBase
from ..services.llm_client import LLMClient
from ..services.lead_service import LeadService
from ..services.telegram_service import (
    send_lead_notification,
    send_support_notification,
    send_human_transfer_notification,
    send_transcript_document,
)
from ..services.validators import is_valid_email, is_valid_phone

# ---------------------------------------------------------------------------
# NOTE: This file was refactored to v2 flow (intent-first, user-friendly).
# Stages: open → (consultation|support|human|qa) → lead_capture
# ---------------------------------------------------------------------------


bp = Blueprint("chat", __name__)

_store = InMemoryConversationStore(max_messages=30)


# ---------------------------------------------------------------------------
# Business type definitions
# ---------------------------------------------------------------------------

_BUSINESS_TYPES: list[tuple[str, set[str]]] = [
    ("Restaurant", {"restaurant", "restaurante"}),
    ("Cafenea", {"cafenea", "cafenele", "cafe", "coffee"}),
    ("Bar / Pub", {"bar", "pub", "bauturi", "băuturi", "cocktail"}),
    ("Fast-food", {"fastfood", "fast-food", "fast", "shaorma", "pizza", "burger", "takeaway"}),
    ("Delivery / Takeaway", {"delivery", "livrare", "to-go", "togo", "comenzi"}),
    ("Lanț de locații", {"lant", "lanț", "multi", "multilocatie", "franciza", "franciză", "locatii", "locații"}),
]

# Business types where tables question is relevant
_TABLES_RELEVANT = {"Restaurant", "Cafenea", "Bar / Pub", "Fast-food"}


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _detect_yes(text: str) -> bool:
    t = _normalize(text)
    words = t.split()
    if len(words) > 4:
        return False
    return any(p in words for p in ["da", "sigur", "ok", "bine", "desigur", "vreau", "doresc"]) or t in {"yes", "y"}


def _detect_no(text: str) -> bool:
    t = _normalize(text)
    words = t.split()
    if len(words) > 4:
        return False
    return any(p in words for p in ["nu", "nici", "nup", "no"]) or t in {"n"}


def _classify_yes_no(text: str, api_key: str, model: str) -> str:
    """LLM-based yes/no classifier. Returns 'YES', 'NO', or 'UNKNOWN'."""
    if not api_key:
        return "UNKNOWN"
    classifier = LLMClient(api_key=api_key, model=model)
    messages = [
        {"role": "system", "content": (
            "Ești un clasificator STRICT. "
            "Contextul: botul RSistems tocmai a întrebat utilizatorul: '’Doriți să vă contacteze un consultant RSistems? (da / nu)'. "
            "Determină dacă mesajul utilizatorului este EXPLICIT un răspuns POZITIV la această întrebare (confirmă clar că vrea să fie contactat) "
            "sau EXPLICIT NEGATIV (refuză clar contactul). "
            "Dacă mesajul este o întrebare, un comentariu, un subiect diferit sau există ORICE dubiu, răspunde cu UNKNOWN. "
            "Răspunde DOAR cu: YES | NO | UNKNOWN. Fără text suplimentar."
        )},
        {"role": "user", "content": text},
    ]
    try:
        result = classifier.chat(messages=messages).strip().upper()
        if result in {"YES", "NO", "UNKNOWN"}:
            return result
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def _extract_business_type(text: str) -> str | None:
    t = _normalize(text)
    for label, keywords in _BUSINESS_TYPES:
        for k in keywords:
            if k in t:
                return label
    return None


def _detect_support_intent(text: str) -> bool:
    t = _normalize(text)
    keywords = ["nu merge", "nu functioneaza", "nu funcționează", "problema", "problemă",
                "eroare", "suport", "defect", "stricat", "nu porneste", "nu pornește",
                "nu tipareste", "nu tipărește", "bon", "casa de marcat", "pos nu"]
    return any(k in t for k in keywords)


def _detect_human_intent(text: str) -> bool:
    t = _normalize(text)
    keywords = ["manager", "operator", "persoana", "persoană", "consultant",
                "om", "angajat", "vorbesc cu", "vorbi cu", "transfer"]
    return any(k in t for k in keywords)


def _detect_demo_intent(text: str) -> bool:
    t = _normalize(text)
    if "demo" in t or "demonstr" in t:
        return True
    if "vreau" in t and ("sa vad" in t or "să văd" in t or "o prezentare" in t):
        return True
    return False


def _demo_negated(text: str) -> bool:
    t = _normalize(text)
    if any(p in t for p in ["nu vreau", "nu doresc", "nu acum", "nu multumesc",
                             "nu, multumesc", "nu mersi", "fara demo", "fără demo",
                             "nu sunt interesat", "nu sunt interesata"]):
        return True
    if t in {"nu", "nu.", "nu!", "no", "n"}:
        return True
    return False


def _extract_email(text: str) -> str | None:
    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return m.group(0) if m else None


def _last_assistant_message(history: list[dict]) -> str | None:
    for m in reversed(history):
        if m.get("role") == "assistant":
            return m.get("content")
    return None


def _assistant_asked_demo(text: str | None) -> bool:
    if not text:
        return False
    t = _normalize(text)
    if "demo" not in t:
        return False
    if "?" in text:
        return True
    return any(k in t for k in ["doriti", "doriți", "vreti", "vrei", "program"])


def _strip_leading_padding(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = text.lstrip()
    for _ in range(2):
        new_out = re.sub(
            r"^(Înțeleg|Inteleg|Sigur|Desigur|Bine|Perfect|Ok|Okay|În regulă|In regulă)\s*[\.!,:;\-–]\s+",
            "", out, flags=re.IGNORECASE,
        )
        if new_out == out:
            break
        out = new_out.lstrip()
    return out


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def _system_prompt() -> str:
    return (
        "Ești un consultant de vânzări pentru RSistems, o companie din România care oferă soluții de automatizare "
        "pentru HoReCa (restaurante, cafenele, baruri, fast-food, delivery), retail, supraveghere video, "
        "sisteme de parcare și panouri digitale.\n\n"
        "Scopul tău:\n"
        "- Înțelegi afacerea clientului\n"
        "- Identifici nevoile principale\n"
        "- Recomanzi soluția potrivită RSistems\n"
        "- Ghidezi clientul spre a solicita o ofertă/demo\n\n"
        "Stil:\n"
        "- Profesional, clar, consultativ — nu agresiv\n"
        "- Răspunsuri scurte, 2–4 propoziții\n"
        "- Fără introduceri de tipul «Înțeleg», «Sigur», «Desigur»\n"
        "- La întrebări despre prețuri, citezi prețurile din baza de cunoștințe (Basic 39€/lună, Professional 59€/lună, Enterprise 89€/lună); pentru echipamente hardware sau ofertă completă propui contact cu un consultant\n"
        "- Nu oferi diagnostic tehnic avansat fără date\n"
        "- Vorbești mereu în română\n\n"
        "Dacă informația nu este în baza de cunoștințe, spui că nu poți răspunde și întrebi dacă dorește să fie contactat de un consultant."
    )


def _recommendation_prompt(business_type: str, locations: int, tables: int | None,
                            existing: bool | None, city: str) -> str:
    existing_str = "da" if existing is True else ("nu" if existing is False else "necunoscut")
    tables_str = f"{tables} mese/puncte de vânzare" if tables else "necunoscut"
    return (
        f"Clientul are o afacere de tip {business_type}, cu {locations} locație/locații, "
        f"{tables_str}, sistem existent: {existing_str}, oraș: {city or 'necunoscut'}.\n\n"
        "Pe baza informațiilor de mai sus și a bazei de cunoștințe RSistems, "
        "scrie o recomandare concisă (3–5 propoziții) cu soluțiile potrivite. "
        "Nu adăuga nicio întrebare la final — aceasta va fi adăugată automat."
    )


def _greeting() -> str:
    return (
        "Bună! Sunt asistentul virtual RSistems.\n"
        "Vă pot ajuta să găsiți soluția potrivită pentru afacerea dvs., să rezolvați o problemă tehnică "
        "sau să vă conectez cu un consultant RSistems.\n\n"
        "Cu ce vă pot ajuta astăzi?"
    )


def _clarifying_question() -> str:
    return (
        "Puteți să îmi spuneți mai multe? "
        "Căutați o soluție pentru afacerea dvs., aveți o problemă tehnică "
        "sau doriți să vorbiți direct cu un consultant RSistems?"
    )


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm_reply(*, user_text: str, history: list[dict], kb_snippets: list[str],
               api_key: str, model: str, extra_instruction: str = "") -> str:
    kb_context = "\n\n".join(kb_snippets) if kb_snippets else ""
    system = _system_prompt()
    if kb_context:
        system += "\n\nBază de cunoștințe (RSistems):\n" + kb_context
    if extra_instruction:
        system += "\n\n" + extra_instruction

    messages = [{"role": "system", "content": system}] + [
        m for m in history if m.get("role") != "system"
    ]
    client = LLMClient(api_key=api_key, model=model)
    return _strip_leading_padding(client.chat(messages=messages))


def _classify_demo_intent_llm(*, user_text: str, last_assistant: str | None,
                               api_key: str, model: str) -> str:
    if not api_key:
        return "UNKNOWN"
    classifier = LLMClient(api_key=api_key, model=model)
    messages = [
        {"role": "system", "content": (
            "Ești un clasificator STRICT de intenție. "
            "Răspunde DOAR cu una din etichetele: WANTS_DEMO | DOESNT_WANT_DEMO | UNKNOWN. "
            "Nu adăuga text suplimentar."
        )},
        {"role": "user", "content": (
            f"Ultimul mesaj asistent: {last_assistant or ''}\n"
            f"Mesaj utilizator: {user_text}\nEtichetă:"
        )},
    ]
    label = (classifier.chat(messages=messages, temperature=0.0) or "").strip().upper()
    label = re.sub(r"[^A-Z_]+", "", label)
    return label if label in {"WANTS_DEMO", "DOESNT_WANT_DEMO", "UNKNOWN"} else "UNKNOWN"


def _classify_demo_intent(*, user_text: str, last_assistant: str | None,
                          api_key: str, model: str) -> str:
    assistant_asked = _assistant_asked_demo(last_assistant)
    if assistant_asked and _detect_no(user_text):
        return "DOESNT_WANT_DEMO"
    if assistant_asked and _detect_yes(user_text) and not _demo_negated(user_text):
        return "WANTS_DEMO"
    if _demo_negated(user_text):
        return "DOESNT_WANT_DEMO"
    if _detect_demo_intent(user_text) and not _demo_negated(user_text):
        return "WANTS_DEMO"
    needs_llm = (
        assistant_asked
        or any(k in _normalize(user_text) for k in ["demo", "demonstr", "prezentare", "sa vad", "să văd"])
    )
    if needs_llm:
        return _classify_demo_intent_llm(
            user_text=user_text, last_assistant=last_assistant,
            api_key=api_key, model=model,
        )
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Path intent classifier (open stage)
# ---------------------------------------------------------------------------

_INTENT_LABELS = {"CONSULTATION", "SUPPORT", "HUMAN", "QA", "UNKNOWN"}


def _classify_path_intent_rules(text: str) -> str | None:
    """Fast rule-based pre-filter for high-confidence signals only.

    Only returns SUPPORT or HUMAN — CONSULTATION vs QA is left to the LLM
    because keyword overlap (e.g. 'sistem' inside 'rsistems') causes false positives.
    Returns None to fall through to LLM for everything else.
    """
    t = _normalize(text)

    # Support: explicit broken/not-working phrases
    support_phrases = [
        "nu merge", "nu functioneaza", "nu funcționează",
        "nu lucreaza", "nu lucrează", "nu mai lucreaza", "nu mai lucrează",
        "nu porneste", "nu pornește", "nu tipareste", "nu tipărește",
        "nu raspunde", "nu răspunde", "nu afiseaza", "nu afișează",
        "casa de marcat nu", "pos nu", "bon fiscal",
        "eroare", "defect", "stricat",
    ]
    if any(k in t for k in support_phrases):
        return "SUPPORT"

    # Support: negation word + hardware/device name in same message
    negation = any(w in t.split() for w in ["nu", "n-a", "n-am"])
    device_kw = ["pos", "casa de marcat", "bon", "imprimanta", "imprimantă",
                 "ecran", "display", "scanner", "sertar", "terminal", "chitanta", "chitanță"]
    if negation and any(k in t for k in device_kw):
        return "SUPPORT"

    # Human: explicit request for a person/manager
    human_kw = ["manager", "operator", "om real", "persoana", "persoană",
                "vorbesc cu", "vorbi cu", "transfer", "angajat",
                "vreau sa vorbesc", "vreau să vorbesc"]
    if any(k in t for k in human_kw):
        return "HUMAN"

    # Everything else (CONSULTATION vs QA) — let the LLM decide
    return None


def _classify_path_intent_llm(user_text: str, api_key: str, model: str) -> str:
    """LLM classifier for path intent when rules are insufficient."""
    if not api_key:
        return "UNKNOWN"
    classifier = LLMClient(api_key=api_key, model=model)
    messages = [
        {"role": "system", "content": (
            "Ești un clasificator de intenție pentru un chatbot de vânzări RSistems (România). "
            "Clasifică mesajul utilizatorului în una din categoriile:\n"
            "CONSULTATION - vrea o soluție/ofertă/demo pentru afacerea lui\n"
            "SUPPORT - are o problemă tehnică cu echipamente/soft existente\n"
            "HUMAN - vrea să vorbească cu un om/manager/consultant\n"
            "QA - are o întrebare despre RSistems, produse sau servicii\n"
            "UNKNOWN - mesaj ambiguu sau salut fără context\n"
            "Răspunde DOAR cu eticheta, fără text suplimentar."
        )},
        {"role": "user", "content": f"Mesaj: {user_text}\nEtichetă:"},
    ]
    label = (classifier.chat(messages=messages, temperature=0.0) or "").strip().upper()
    label = re.sub(r"[^A-Z]+", "", label)
    return label if label in _INTENT_LABELS else "UNKNOWN"


def _classify_path_intent(user_text: str, api_key: str, model: str) -> str:
    """Rules-first, LLM fallback path intent classifier."""
    rule = _classify_path_intent_rules(user_text)
    if rule:
        return rule
    return _classify_path_intent_llm(user_text, api_key, model)


# ---------------------------------------------------------------------------
# KB helper
# ---------------------------------------------------------------------------

def _kb_search(user_text: str) -> list[str]:
    project_root = os.path.abspath(os.path.join(current_app.root_path, os.pardir))
    kb = KnowledgeBase(
        kb_dir=os.path.join(project_root, "kb"),
        index_path=os.path.join(project_root, "kb_index.json"),
        openai_api_key=current_app.config.get("OPENAI_API_KEY", ""),
        embedding_model=current_app.config.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    return kb.search(user_text)


# ---------------------------------------------------------------------------
# Flow handlers — each returns a (reply, meta_updates) tuple or None
# ---------------------------------------------------------------------------

def _handle_qualifying(conversation_id: str, user_text: str, meta: dict) -> str | None:
    """Handle the multi-step qualifying stage. Returns reply or None if stage not active."""
    q = meta.get("qualifying") or {}
    step = q.get("step")
    if not step:
        return None

    business_type = meta.get("business_type", "")

    if step == "locations":
        if re.search(r"\d+\s*[\.,]\s*\d+", user_text):
            return "Vă rog să îmi spuneți un număr întreg de locații (ex: 1, 2, 3)."
        m = re.search(r"\d+", user_text)
        locations = int(m.group(0)) if m else 0
        if locations < 1:
            return "Vă rog să îmi spuneți un număr valid de locații (minim 1)."
        q["locations"] = locations
        # Decide next step
        if business_type in _TABLES_RELEVANT:
            q["step"] = "tables"
            _store.update_meta(conversation_id, {"qualifying": q})
            return "Câte mese (sau puncte de vânzare) are locația dvs.?"
        else:
            q["step"] = "existing"
            _store.update_meta(conversation_id, {"qualifying": q})
            return "Aveți deja un sistem POS sau de gestiune în funcțiune? (da / nu)"

    if step == "tables":
        m = re.search(r"\d+", user_text)
        tables = int(m.group(0)) if m else None
        q["tables"] = tables
        q["step"] = "existing"
        _store.update_meta(conversation_id, {"qualifying": q})
        return "Aveți deja un sistem POS sau de gestiune în funcțiune? (da / nu)"

    if step == "existing":
        if _detect_yes(user_text):
            q["existing"] = True
        elif _detect_no(user_text):
            q["existing"] = False
        else:
            return "Vă rog să răspundeți cu da sau nu — aveți deja un sistem POS/gestiune?"
        q["step"] = "city"
        _store.update_meta(conversation_id, {"qualifying": q})
        return "În ce oraș/localitate vă aflați?"

    if step == "city":
        city = user_text.strip()
        if len(city) < 2:
            return "Vă rog să îmi spuneți orașul (minim 2 caractere)."
        q["city"] = city
        q["step"] = "done"
        _store.update_meta(conversation_id, {"qualifying": q, "stage": "recommendation"})
        # Trigger recommendation on next turn — return sentinel
        return "__RECOMMENDATION__"

    return None


def _handle_support_flow(conversation_id: str, user_text: str, meta: dict) -> str | None:
    """Handle multi-step support ticket collection. Returns reply or None."""
    s = meta.get("support") or {}
    step = s.get("step")
    if not step:
        return None

    if step == "company":
        company = user_text.strip()
        if len(company) < 2:
            return "Vă rog să îmi spuneți numele companiei (minim 2 caractere)."
        s["company"] = company
        s["step"] = "contact_name"
        _store.update_meta(conversation_id, {"support": s})
        return "Cum vă numiți (persoana de contact)?"

    if step == "contact_name":
        name = user_text.strip()
        if len(name) < 2:
            return "Vă rog să îmi spuneți numele dvs. (minim 2 caractere)."
        s["contact_name"] = name
        s["step"] = "phone"
        _store.update_meta(conversation_id, {"support": s})
        return "Îmi lăsați un număr de telefon pentru a vă contacta rapid?"

    if step == "phone":
        if not is_valid_phone(user_text.strip()):
            return "Nu am recunoscut un număr valid. Vă rog să îl scrieți din nou (ex: 07xx xxx xxx)."
        s["phone"] = user_text.strip()
        s["step"] = "issue"
        _store.update_meta(conversation_id, {"support": s})
        return "Descrieți pe scurt problema cu care vă confruntați."

    if step == "issue":
        issue = user_text.strip()
        if len(issue) < 5:
            return "Vă rog să descrieți puțin mai detaliat problema."
        s["issue"] = issue
        s["step"] = None
        _store.update_meta(conversation_id, {"support": s, "stage": "ended"})

        try:
            send_support_notification(
                company=s.get("company", ""),
                contact_name=s.get("contact_name", ""),
                phone=s.get("phone", ""),
                issue=issue,
            )
        except Exception as exc:
            current_app.logger.warning("Support Telegram notification failed: %s", exc)

        try:
            history = _store.get(conversation_id)
            send_transcript_document(
                lead_ref=f"Suport — {s.get('company', 'necunoscut')}",
                messages=history,
            )
        except Exception as exc:
            current_app.logger.warning("Support transcript send failed: %s", exc)

        return (
            "Am înregistrat solicitarea de suport. Un specialist RSistems vă va contacta cât mai curând.\n\n"
            "Vă mulțumim că ați contactat RSistems. O zi bună!"
        )

    return None


def _handle_human_transfer(conversation_id: str, user_text: str, meta: dict) -> str | None:
    """Handle multi-step human transfer collection. Returns reply or None."""
    h = meta.get("human_transfer") or {}
    step = h.get("step")
    if not step:
        return None

    if step == "name":
        name = user_text.strip()
        if len(name) < 2:
            return "Vă rog să îmi spuneți numele dvs. (minim 2 caractere)."
        h["name"] = name
        h["step"] = "phone"
        _store.update_meta(conversation_id, {"human_transfer": h})
        return "Îmi lăsați numărul de telefon?"

    if step == "phone":
        if not is_valid_phone(user_text.strip()):
            return "Nu am recunoscut un număr valid. Vă rog să îl scrieți din nou (ex: 07xx xxx xxx)."
        h["phone"] = user_text.strip()
        h["step"] = "topic"
        _store.update_meta(conversation_id, {"human_transfer": h})
        return "Cu ce subiect doriți să vorbiți cu managerul? (ex: ofertă, demo, contract, altul)"

    if step == "topic":
        topic = user_text.strip()
        if len(topic) < 2:
            return "Vă rog să specificați subiectul (minim 2 caractere)."
        h["topic"] = topic
        h["step"] = None
        _store.update_meta(conversation_id, {"human_transfer": h, "stage": "ended"})

        try:
            send_human_transfer_notification(
                name=h.get("name", ""),
                phone=h.get("phone", ""),
                topic=topic,
            )
        except Exception as exc:
            current_app.logger.warning("Human transfer Telegram notification failed: %s", exc)

        try:
            history = _store.get(conversation_id)
            send_transcript_document(
                lead_ref=f"Transfer — {h.get('name', 'necunoscut')}",
                messages=history,
            )
        except Exception as exc:
            current_app.logger.warning("Human transfer transcript send failed: %s", exc)

        return (
            "Am transmis solicitarea dvs. Un manager RSistems vă va contacta în cel mai scurt timp.\n\n"
            "Vă mulțumim că ați contactat RSistems. O zi bună!"
        )

    return None


def _handle_lead_capture(conversation_id: str, user_text: str, meta: dict) -> str | None:
    """Handle multi-step lead contact collection. Returns reply or None."""
    lead_state = meta.get("lead") or {}
    if not lead_state.get("active"):
        return None

    step = lead_state.get("step")
    draft = dict(lead_state.get("draft") or {})

    if step == "name":
        name = user_text.strip()
        if len(name) < 2:
            return "Vă rog să îmi spuneți numele dvs. (minim 2 caractere)."
        draft["name"] = name
        lead_state.update({"step": "phone", "draft": draft})
        _store.update_meta(conversation_id, {"lead": lead_state})
        return "Îmi lăsați, vă rog, numărul de telefon? (ex: 07xx xxx xxx / +40...)"

    if step == "phone":
        if not is_valid_phone(user_text.strip()):
            return "Nu am recunoscut un număr valid. Vă rog să îl scrieți din nou (ex: 07xx xxx xxx / +40...)."
        draft["phone"] = user_text.strip()
        lead_state.update({"step": "email", "draft": draft})
        _store.update_meta(conversation_id, {"lead": lead_state})
        return "Îmi lăsați și adresa de email? (ex: nume@domeniu.ro)"

    if step == "email":
        t = _normalize(user_text)
        if any(k in t for k in ["nu am", "nu am email", "nu", "skip", "fara", "fără"]):
            draft["email"] = None
        else:
            email = _extract_email(user_text)
            if not email or not is_valid_email(email):
                return "Nu am recunoscut un email valid. Vă rog să îl scrieți (ex: nume@domeniu.ro) sau scrieți 'nu am'."
            draft["email"] = email
        if not meta.get("business_type"):
            lead_state.update({"step": "business_type_q", "draft": draft})
            _store.update_meta(conversation_id, {"lead": lead_state})
            return (
                "Pentru ce tip de afacere căutați o soluție? "
                "(Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații)"
            )
        lead_state.update({"step": "city_q", "draft": draft})
        _store.update_meta(conversation_id, {"lead": lead_state})
        return "În ce oraș/localitate vă aflați?"

    if step == "business_type_q":
        bt = _extract_business_type(user_text)
        if not bt:
            return (
                "Nu am recunoscut tipul afacerii. Vă rog alegeți: "
                "Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații"
            )
        draft["business_type"] = bt
        _store.update_meta(conversation_id, {"business_type": bt})
        lead_state.update({"step": "city_q", "draft": draft})
        _store.update_meta(conversation_id, {"lead": lead_state})
        return "În ce oraș/localitate vă aflați?"

    if step == "city_q":
        city = user_text.strip()
        draft["city"] = city if len(city) >= 2 else None
        lead_state.update({"step": "business_name", "draft": draft})
        _store.update_meta(conversation_id, {"lead": lead_state})
        return "Care este numele afacerii/locației dvs.?"

    if step == "business_name":
        business_name = user_text.strip()
        if len(business_name) < 2:
            return "Vă rog să îmi spuneți numele afacerii (minim 2 caractere)."
        draft["business_name"] = business_name
        lead_state.update({"step": None, "active": False, "draft": draft})
        _store.update_meta(conversation_id, {"lead": lead_state, "stage": "ended"})

        q = meta.get("qualifying") or {}
        payload = {
            "name": draft.get("name"),
            "phone": draft.get("phone"),
            "email": draft.get("email"),
            "business_name": business_name,
            "type_of_business": meta.get("business_type") or draft.get("business_type"),
            "nr_of_locations": q.get("locations"),
            "tables_count": q.get("tables"),
            "has_existing_system": q.get("existing"),
            "city": q.get("city") or draft.get("city"),
        }

        try:
            lead = LeadService.create_lead(payload)
        except Exception as exc:
            current_app.logger.warning("Lead creation failed: %s", exc)
            _store.update_meta(conversation_id, {"stage": "lead_capture"})
            return (
                "A apărut o problemă la salvarea datelor. "
                "Puteți încerca din nou sau ne lăsați doar un număr de telefon și revenim noi."
            )

        try:
            send_lead_notification(lead)
        except Exception as exc:
            current_app.logger.warning("Telegram lead notification failed: %s", exc)

        try:
            history = _store.get(conversation_id)
            send_transcript_document(
                lead_ref=f"Lead #{lead.id}",
                messages=history,
            )
        except Exception as exc:
            current_app.logger.warning("Lead transcript send failed: %s", exc)

        return (
            "Mulțumesc! Am înregistrat solicitarea. "
            "Un consultant RSistems vă va contacta în cel mai scurt timp.\n\n"
            "Vă mulțumim că ați contactat RSistems. O zi bună!"
        )

    return None


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------

@bp.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    conversation_id = payload.get("conversation_id")
    message = payload.get("message")

    # First call: create conversation and greet
    if not conversation_id:
        conversation_id = _store.create(initial_messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "assistant", "content": _greeting()},
        ])
        _store.update_meta(conversation_id, {
            "stage": "open",
            "business_type": None,
            "qualifying": {"step": None},
            "support": {"step": None},
            "human_transfer": {"step": None},
            "lead": {"active": False, "step": None, "draft": {}},
            "clarify_attempted": False,
            "pending_contact_confirm": False,
        })
        return jsonify({"conversation_id": conversation_id, "reply": _greeting()})

    if not _store.exists(conversation_id):
        return jsonify({
            "error": "unknown_conversation",
            "message": "Unknown conversation_id. Start a new conversation without conversation_id.",
        }), 400

    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "invalid_request", "message": "'message' is required."}), 400

    user_text = message.strip()
    meta = _store.get_meta(conversation_id)
    stage = meta.get("stage")

    # Conversation closed — ignore further messages
    if stage == "ended":
        return jsonify({
            "conversation_id": conversation_id,
            "reply": "Această conversație s-a încheiat. Scrieți /new pentru a începe una nouă.",
        })

    api_key = current_app.config.get("OPENAI_API_KEY", "")
    model = current_app.config.get("OPENAI_MODEL", "gpt-4o-mini")

    def _respond(reply: str) -> object:
        _store.append(conversation_id, {"role": "assistant", "content": reply})
        return jsonify({"conversation_id": conversation_id, "reply": reply})

    # ── Active sub-flows (highest priority — run before stage checks) ─────────

    support_reply = _handle_support_flow(conversation_id, user_text, meta)
    if support_reply is not None:
        _store.append(conversation_id, {"role": "user", "content": user_text})
        return _respond(support_reply)

    transfer_reply = _handle_human_transfer(conversation_id, user_text, meta)
    if transfer_reply is not None:
        _store.append(conversation_id, {"role": "user", "content": user_text})
        return _respond(transfer_reply)

    lead_reply = _handle_lead_capture(conversation_id, user_text, meta)
    if lead_reply is not None:
        _store.append(conversation_id, {"role": "user", "content": user_text})
        return _respond(lead_reply)

    # ── Stage: open — classify intent from first free-text message ────────────
    if stage == "open":
        intent = _classify_path_intent(user_text, api_key, model)

        if intent == "SUPPORT":
            _store.update_meta(conversation_id, {"support": {"step": "company"}, "stage": "support_flow"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Îmi pare rău că întâmpinați probleme. Pentru a vă ajuta rapid, îmi spuneți numele companiei/locației?")

        if intent == "HUMAN":
            _store.update_meta(conversation_id, {"human_transfer": {"step": "name"}, "stage": "human_transfer"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Sigur. Cum vă numiți, vă rog?")

        if intent == "CONSULTATION":
            # Check if business type already known from message
            bt = _extract_business_type(user_text)
            if bt:
                _store.update_meta(conversation_id, {
                    "business_type": bt,
                    "stage": "qualifying",
                    "qualifying": {"step": "locations"},
                })
                _store.append(conversation_id, {"role": "user", "content": user_text})
                return _respond(f"Am notat: {bt}. Câte locații aveți?")
            else:
                _store.update_meta(conversation_id, {"stage": "awaiting_business_type"})
                _store.append(conversation_id, {"role": "user", "content": user_text})
                return _respond(
                    "Cu plăcere! Pentru ce tip de afacere căutați o soluție? "
                    "(Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații)"
                )

        if intent == "QA":
            _store.update_meta(conversation_id, {"stage": "chatting"})
            # Fall through to KB+LLM section below
            stage = "chatting"

        if intent == "UNKNOWN":
            if not meta.get("clarify_attempted"):
                _store.update_meta(conversation_id, {"clarify_attempted": True})
                _store.append(conversation_id, {"role": "user", "content": user_text})
                return _respond(_clarifying_question())
            else:
                # Second attempt still unknown → default to QA / chatting
                _store.update_meta(conversation_id, {"stage": "chatting", "clarify_attempted": False})
                stage = "chatting"

    # ── Stage: awaiting_business_type ─────────────────────────────────────────
    if stage == "awaiting_business_type":
        if _detect_support_intent(user_text):
            _store.update_meta(conversation_id, {"support": {"step": "company"}, "stage": "support_flow"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Îmi pare rău că întâmpinați probleme. Îmi spuneți numele companiei/locației?")

        if _detect_human_intent(user_text):
            _store.update_meta(conversation_id, {"human_transfer": {"step": "name"}, "stage": "human_transfer"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Sigur. Cum vă numiți, vă rog?")

        bt = _extract_business_type(user_text)
        if not bt:
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond(
                "Nu am reușit să identific tipul afacerii. "
                "Lucrați cu un Restaurant, Cafenea, Bar-Pub, Fast-food, Delivery sau Lanț de locații?"
            )

        _store.update_meta(conversation_id, {
            "business_type": bt,
            "stage": "qualifying",
            "qualifying": {"step": "locations"},
        })
        _store.append(conversation_id, {"role": "user", "content": user_text})
        return _respond(f"Am notat: {bt}. Câte locații aveți?")

    # ── Stage: qualifying ─────────────────────────────────────────────────────
    if stage == "qualifying":
        q_reply = _handle_qualifying(conversation_id, user_text, meta)
        if q_reply == "__RECOMMENDATION__":
            meta = _store.get_meta(conversation_id)
            q = meta.get("qualifying") or {}
            business_type = meta.get("business_type", "")
            _store.append(conversation_id, {"role": "user", "content": user_text})

            try:
                snippets = _kb_search(business_type)
            except Exception:
                snippets = []

            extra = _recommendation_prompt(
                business_type=business_type,
                locations=q.get("locations", 1),
                tables=q.get("tables"),
                existing=q.get("existing"),
                city=q.get("city", ""),
            )
            history = _store.get(conversation_id)
            try:
                rec_reply = _llm_reply(
                    user_text=user_text, history=history, kb_snippets=snippets,
                    api_key=api_key, model=model, extra_instruction=extra,
                )
            except Exception as exc:
                current_app.logger.warning("LLM recommendation failed: %s", exc)
                rec_reply = (
                    f"Pe baza informațiilor colectate, vă pot recomanda o soluție completă RSistems pentru {business_type}. "
                    "Un consultant vă poate pregăti o ofertă personalizată."
                )
            if "(da / nu)" not in rec_reply and "(da/nu)" not in rec_reply:
                rec_reply = rec_reply.rstrip() + "\n\nDoriți să vă contacteze un consultant RSistems? (da / nu)"
            _store.update_meta(conversation_id, {"stage": "pending_contact_confirm"})
            return _respond(rec_reply)

        if q_reply is not None:
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond(q_reply)

    # ── Stage: pending_contact_confirm ────────────────────────────────────────
    if stage == "pending_contact_confirm":
        yn = _classify_yes_no(user_text, api_key, model)
        if yn == "YES":
            _store.update_meta(conversation_id, {
                "lead": {"active": True, "step": "name", "draft": {}},
                "stage": "lead_capture",
            })
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Cum vă numiți, vă rog?")
        if yn == "NO":
            _store.update_meta(conversation_id, {"stage": "chatting"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Înțeles. Dacă aveți întrebări suplimentare, sunt aici să vă ajut.")
        # Not a yes/no — answer the question first, then re-ask
        try:
            snippets = _kb_search(user_text)
        except Exception:
            snippets = []
        _store.append(conversation_id, {"role": "user", "content": user_text})
        history = _store.get(conversation_id)
        try:
            llm_answer = _llm_reply(
                user_text=user_text, history=history, kb_snippets=snippets,
                api_key=api_key, model=model,
            )
        except Exception:
            llm_answer = "Nu am putut găsi un răspuns exact la întrebarea dvs."
        return _respond(llm_answer + "\n\nDoriți să vă contacteze un consultant RSistems? (da / nu)")

    # ── Stage: chatting (free KB + LLM) ──────────────────────────────────────
    if stage == "chatting":
        if _detect_support_intent(user_text):
            _store.update_meta(conversation_id, {"support": {"step": "company"}, "stage": "support_flow"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Îmi pare rău că întâmpinați probleme. Îmi spuneți numele companiei/locației?")

        if _detect_human_intent(user_text):
            _store.update_meta(conversation_id, {"human_transfer": {"step": "name"}, "stage": "human_transfer"})
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Sigur. Cum vă numiți, vă rog?")

        # Re-classify to catch consultation intent during free chat.
        # Skip if qualifying already done — follow-up questions should go to LLM.
        qualifying_done = meta.get("qualifying", {}).get("step") == "done"
        if not qualifying_done:
            chat_intent = _classify_path_intent(user_text, api_key, model)
            if chat_intent == "CONSULTATION":
                bt = _extract_business_type(user_text) or meta.get("business_type")
                if bt:
                    _store.update_meta(conversation_id, {
                        "business_type": bt,
                        "stage": "qualifying",
                        "qualifying": {"step": "locations"},
                    })
                    _store.append(conversation_id, {"role": "user", "content": user_text})
                    return _respond(f"Am notat: {bt}. Câte locații aveți?")
                else:
                    _store.update_meta(conversation_id, {"stage": "awaiting_business_type"})
                    _store.append(conversation_id, {"role": "user", "content": user_text})
                    return _respond(
                        "Cu plăcere! Pentru ce tip de afacere căutați o soluție? "
                        "(Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații)"
                    )

    history = _store.get(conversation_id)
    last_assistant = _last_assistant_message(history)

    # Demo intent shortcut — always routes through qualifying if not yet done
    if _assistant_asked_demo(last_assistant) or _detect_demo_intent(user_text):
        label = _classify_demo_intent(
            user_text=user_text, last_assistant=last_assistant,
            api_key=api_key, model=model,
        )
        if label == "WANTS_DEMO":
            qualifying_done = meta.get("qualifying", {}).get("step") == "done"
            if qualifying_done:
                _store.update_meta(conversation_id, {
                    "lead": {"active": True, "step": "name", "draft": {}},
                    "stage": "lead_capture",
                })
                _store.append(conversation_id, {"role": "user", "content": user_text})
                return _respond("Super! Cum vă numiți, vă rog?")
            else:
                bt = _extract_business_type(user_text) or meta.get("business_type")
                if bt:
                    _store.update_meta(conversation_id, {
                        "business_type": bt,
                        "stage": "qualifying",
                        "qualifying": {"step": "locations"},
                    })
                    _store.append(conversation_id, {"role": "user", "content": user_text})
                    return _respond(f"Super! Înainte de demo, am nevoie de câteva detalii. Câte locații aveți?")
                else:
                    _store.update_meta(conversation_id, {"stage": "awaiting_business_type"})
                    _store.append(conversation_id, {"role": "user", "content": user_text})
                    return _respond(
                        "Super! Înainte de demo, pentru ce tip de afacere căutați o soluție? "
                        "(Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații)"
                    )
        if label == "DOESNT_WANT_DEMO":
            _store.append(conversation_id, {"role": "user", "content": user_text})
            return _respond("Spuneți-mi ce ați dori să aflați despre sistem.")

    # KB + LLM
    try:
        snippets = _kb_search(user_text)
    except ValueError as exc:
        return jsonify({"error": "config_error", "message": str(exc)}), 500

    _store.append(conversation_id, {"role": "user", "content": user_text})
    history = _store.get(conversation_id)

    if not snippets:
        fallback = "Nu sunt sigur că am înțeles. Doriți să vă contactăm? (da/nu)"
        _store.update_meta(conversation_id, {"stage": "pending_contact_confirm"})
        return _respond(fallback)

    try:
        reply = _llm_reply(
            user_text=user_text, history=history, kb_snippets=snippets,
            api_key=api_key, model=model,
        )
    except ValueError as exc:
        return jsonify({"error": "config_error", "message": str(exc)}), 500

    return _respond(reply)
