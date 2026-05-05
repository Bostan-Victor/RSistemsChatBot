from __future__ import annotations

import os
import re

from flask import Blueprint, current_app, jsonify, request

from ..services.conversation_store import InMemoryConversationStore
from ..services.knowledge_base import KnowledgeBase
from ..services.llm_client import LLMClient
from ..services.lead_service import LeadService
from ..services.telegram_service import send_lead_notification
from ..services.validators import is_valid_email, is_valid_phone


bp = Blueprint("chat", __name__)

_store = InMemoryConversationStore(max_messages=30)


_BUSINESS_TYPES: list[tuple[str, set[str]]] = [
	("Restaurant", {"restaurant", "restaurante"}),
	("Cafenea", {"cafenea", "cafenele", "cafe"}),
	("Bar / Pub", {"bar", "pub"}),
	("Fast-food", {"fastfood", "fast-food", "fast", "shaorma", "pizza", "burger"}),
	("Delivery / Takeaway", {"delivery", "livrare", "takeaway", "to-go", "togo", "comenzi"}),
	("Lanț de locații", {"lant", "lanț", "multi", "multilocatie", "franciza", "franciză", "locatii", "locații"}),
]


def _normalize_simple(text: str) -> str:
	return re.sub(r"\s+", " ", text.strip().lower())


def _detect_yes(text: str) -> bool:
	t = _normalize_simple(text)
	return any(p in t.split() for p in ["da", "sigur", "ok", "bine", "desigur", "vreau", "doresc"]) or t in {"yes", "y"}


def _detect_no(text: str) -> bool:
	t = _normalize_simple(text)
	return any(p in t.split() for p in ["nu", "nici", "nup", "no"]) or t in {"n"}


def _detect_demo_intent(text: str) -> bool:
	t = _normalize_simple(text)
	if "demo" in t or "demonstr" in t:
		return True
	if "vreau" in t and ("sa vad" in t or "să văd" in t or "o prezentare" in t):
		return True
	if "program" in t and ("demo" in t or "o prezentare" in t or "o demonstr" in t):
		return True
	return False


def _demo_negated(text: str) -> bool:
	"""Detect common negations around demo intent."""
	t = _normalize_simple(text)
	# Common explicit negations
	if any(p in t for p in [
		"nu vreau",
		"nu doresc",
		"nu acum",
		"nu multumesc",
		"nu, multumesc",
		"nu mersi",
		"fara demo",
		"fără demo",
		"nu sunt interesat",
		"nu sunt interesata",
		"nu sunt interesată",
	]):
		return True
	# Single-word "nu" is a strong negation in confirmation flows
	if t in {"nu", "nu.", "nu!", "no", "n"}:
		return True
	return False


def _needs_demo_intent_classification(user_text: str, last_assistant: str | None) -> bool:
	"""Return True only when message is potentially about a demo/booking."""
	t = _normalize_simple(user_text)
	if _assistant_asked_demo(last_assistant):
		return True
	# user mentions demo or close concepts
	return any(k in t for k in ["demo", "demonstr", "prezentare", "prezentati", "prezentați", "arat", "arata", "arăta", "sa vad", "să văd"]) \
		or _detect_demo_intent(user_text)


def _classify_demo_intent_hybrid(
	*,
	user_text: str,
	last_assistant: str | None,
	api_key: str,
	model: str,
) -> str:
	"""Hybrid intent classifier: rules first, LLM only when ambiguous.

	Returns one of: WANTS_DEMO, DOESNT_WANT_DEMO, UNKNOWN.
	"""
	assistant_asked = _assistant_asked_demo(last_assistant)
	t = _normalize_simple(user_text)

	# 1) Clear rules
	if assistant_asked and _detect_no(user_text):
		return "DOESNT_WANT_DEMO"
	if assistant_asked and _detect_yes(user_text) and not _demo_negated(user_text):
		return "WANTS_DEMO"

	# If user explicitly negates demo intent, treat as DOESNT (when demo is mentioned or assistant asked)
	if _demo_negated(user_text) and (assistant_asked or "demo" in t or "prezentare" in t or "demonstr" in t):
		return "DOESNT_WANT_DEMO"

	# Positive demo intent (but only if not negated)
	if _detect_demo_intent(user_text) and not _demo_negated(user_text):
		return "WANTS_DEMO"

	# 2) If it doesn't look demo-related, don't spend an LLM call
	if not _needs_demo_intent_classification(user_text, last_assistant):
		return "UNKNOWN"

	# 3) LLM fallback for ambiguous cases
	if not api_key:
		return "UNKNOWN"

	classifier = LLMClient(api_key=api_key, model=model)
	msg_last = (last_assistant or "").strip()
	messages = [
		{
			"role": "system",
			"content": (
				"Ești un clasificator STRICT de intenție. "
				"Primești un mesaj de la utilizator (română) și trebuie să decizi dacă vrea să programeze un demo. "
				"Răspunde DOAR cu una din etichetele următoare, exact așa: "
				"WANTS_DEMO | DOESNT_WANT_DEMO | UNKNOWN. "
				"Nu adăuga explicații, semne de punctuație sau text suplimentar."
			),
		},
		{
			"role": "user",
			"content": (
				f"Ultimul mesaj al asistentului: {msg_last}\n"
				f"Mesaj utilizator: {user_text.strip()}\n"
				"Etichetă:"
			),
		},
	]

	label = (classifier.chat(messages=messages, temperature=0.0) or "").strip().upper()
	label = re.sub(r"[^A-Z_]+", "", label)
	if label in {"WANTS_DEMO", "DOESNT_WANT_DEMO", "UNKNOWN"}:
		return label
	return "UNKNOWN"


def _last_assistant_message(history: list[dict[str, str]]) -> str | None:
	for m in reversed(history):
		if m.get("role") == "assistant":
			return m.get("content")
	return None


def _assistant_asked_demo(text: str | None) -> bool:
	if not text:
		return False
	# Heuristic: if assistant mentions demo and it's phrased like a question/invitation.
	t = _normalize_simple(text)
	if "demo" not in t:
		return False
	if "?" in text:
		return True
	return any(k in t for k in ["doriti", "doriți", "vreti", "vrei", "program", "sa va arat", "să vă arăt"])


def _extract_business_type(text: str) -> str | None:
	t = _normalize_simple(text)
	# direct match against keywords
	for label, keywords in _BUSINESS_TYPES:
		for k in keywords:
			if k in t:
				return label
	return None


def _extract_email(text: str) -> str | None:
	m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
	return m.group(0) if m else None


def _extract_phone(text: str) -> str | None:
	# Very permissive: collect digits, allow + and spaces; require at least 9 digits.
	digits = re.sub(r"\D", "", text)
	if len(digits) < 9:
		return None
	# return original trimmed if it contains plus/digits, else formatted digits
	trimmed = text.strip()
	return trimmed if re.search(r"[0-9]", trimmed) else digits


def _strip_leading_padding(text: str) -> str:
	"""Remove common Romanian filler openers (minimal post-processing).

	Only strips when the opener is a standalone interjection followed by punctuation.
	"""
	if not isinstance(text, str):
		return text

	out = text.lstrip()
	# Apply a couple of times in case the model stacks openers like "Ok. Înțeleg. ..."
	for _ in range(2):
		new_out = re.sub(
			r"^(Înțeleg|Inteleg|Sigur|Desigur|Bine|Perfect|Ok|Okay|În regulă|In regulă|În regula|In regula)\s*[\.!,:;\-–]\s+",
			"",
			out,
			flags=re.IGNORECASE,
		)
		if new_out == out:
			break
		out = new_out.lstrip()

	return out


def _system_prompt() -> str:
	return (
		"Ești un consultant de software pentru HoReCa din partea RSistems.\n\n"
		"Scopul tău este să:\n"
		"- înțelegi afacerea clientului\n"
		"- identifici problemele principale\n"
		"- recomanzi soluția potrivită RSistems\n"
		"- ghidezi clientul către solicitarea unui demo\n\n"
		"Stil:\n"
		"- profesional, dar prietenos\n"
		"- clar, fără jargon tehnic\n"
		"- orientat spre soluții, ca un manager real\n"
		"- răspunsuri scurte și ușor de urmărit\n"
		"- începi direct cu răspunsul (fără introduceri de tipul: «Înțeleg», «Sigur», «Desigur», «Bine» etc.)\n"
		"- NU adaugi propoziții generale/umplutură care nu aduc informație nouă\n"
		"- de regulă 2–4 propoziții; apoi cel mult 1 întrebare de clarificare, dacă e util\n\n"
		"Întotdeauna:\n"
		"- vorbești în limba română\n"
		"- pui 1–2 întrebări de clarificare când e util\n"
		"- NU inventezi prețuri; la întrebări despre cost răspunzi că depinde și propui un demo\n\n"
		"Dacă informația nu este disponibilă în baza de cunoștințe furnizată, spui clar că nu poți răspunde la întrebări care nu țin de RSistems și întrebi dacă dorește să fie contactat de un consultant."
	)


def _greeting() -> str:
	return (
		"Bună! Pentru a vă recomanda soluția potrivită, îmi spuneți ce tip de locație aveți? "
		"(Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații)"
	)


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
		_store.update_meta(
			conversation_id,
			{
				"stage": "awaiting_business_type",
				"business_type": None,
				"lead": {"active": False, "step": None, "draft": {}},
				"pending_human_contact_confirm": False,
			},
		)
		return jsonify(
			{
				"conversation_id": conversation_id,
				"reply": _greeting(),
			}
		)

	if not _store.exists(conversation_id):
		return (
			jsonify(
				{
					"error": "unknown_conversation",
					"message": "Unknown conversation_id. Start a new conversation without conversation_id.",
				}
			),
			400,
		)

	if not isinstance(message, str) or not message.strip():
		return (
			jsonify({"error": "invalid_request", "message": "'message' is required."}),
			400,
		)

	user_text = message.strip()
	meta = _store.get_meta(conversation_id)
	stage = meta.get("stage")
	lead_state = meta.get("lead") or {"active": False, "step": None, "draft": {}}

	# Lead capture mini-form (multi-turn)
	if lead_state.get("active"):
		step = lead_state.get("step")
		draft = dict(lead_state.get("draft") or {})

		if step == "name":
			name = user_text.strip()
			if len(name) < 2:
				reply = "Vă rog să îmi spuneți numele (minim 2 caractere)."
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})
			draft["name"] = name
			lead_state["step"] = "phone"
			lead_state["draft"] = draft
			_store.update_meta(conversation_id, {"lead": lead_state})
			reply = "Mulțumesc. Îmi lăsați, vă rog, numărul de telefon pentru a vă contacta? (ex: 07xx xxx xxx / +40...)"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

		if step == "phone":
			candidate_phone = user_text.strip()
			if not is_valid_phone(candidate_phone):
				reply = "Nu am reușit să identific un număr de telefon valid. Îmi puteți scrie, vă rog, un număr (ex: 07xx xxx xxx / +40...)?"
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})
			draft["phone"] = candidate_phone
			lead_state["step"] = "email"
			lead_state["draft"] = draft
			_store.update_meta(conversation_id, {"lead": lead_state})
			reply = "Mulțumesc. Îmi lăsați și emailul? (ex: nume@domeniu.ro)"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

		if step == "email":
			email = _extract_email(user_text)
			if not email or not is_valid_email(email):
				reply = "Nu am reușit să identific un email valid. Îmi puteți scrie, vă rog, emailul (ex: nume@domeniu.ro)?"
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})
			draft["email"] = email
			lead_state["step"] = "business_name"
			lead_state["draft"] = draft
			_store.update_meta(conversation_id, {"lead": lead_state})
			reply = "Mulțumesc. Care este numele afacerii/locației (denumirea comercială)?"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

		if step == "business_name":
			business_name = user_text.strip()
			if len(business_name) < 2:
				reply = "Vă rog să îmi spuneți numele afacerii (minim 2 caractere)."
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})
			draft["business_name"] = business_name
			lead_state["step"] = "locations"
			lead_state["draft"] = draft
			_store.update_meta(conversation_id, {"lead": lead_state})
			reply = "Perfect. Câte locații aveți? (ex: 1, 2, 3)"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

		if step == "locations":
			# Strict integer validation (reject decimals like 2.5)
			if re.search(r"\d+\s*[\.,]\s*\d+", user_text):
				reply = "Vă rog să îmi spuneți un număr întreg de locații (ex: 1, 2, 3)."
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})
			m = re.search(r"\d+", user_text)
			locations = int(m.group(0)) if m else 0
			if locations < 1:
				reply = "Vă rog să îmi spuneți un număr de locații valid (minim 1)."
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})
			draft["nr_of_locations"] = locations
			# Use business type collected earlier (do not re-ask)
			business_type = meta.get("business_type")
			payload_for_lead = {
				"name": draft.get("name"),
				"phone": draft.get("phone"),
				"email": draft.get("email"),
				"business_name": draft.get("business_name"),
				"type_of_business": business_type,
				"nr_of_locations": draft.get("nr_of_locations"),
			}

			try:
				lead = LeadService.create_lead(payload_for_lead)
			except Exception:
				reply = "A apărut o problemă la salvarea datelor. Puteți încerca din nou sau îmi lăsați doar un număr de telefon/email și revenim noi."
				_store.append(conversation_id, {"role": "assistant", "content": reply})
				return jsonify({"conversation_id": conversation_id, "reply": reply})

			try:
				send_lead_notification(lead)
			except Exception as exc:
				current_app.logger.warning("Telegram notification failed: %s", exc)

			# Reset lead capture
			_store.update_meta(
				conversation_id,
				{
					"lead": {"active": False, "step": None, "draft": {}},
					"pending_human_contact_confirm": False,
				},
			)
			reply = (
				"Mulțumesc! Am înregistrat solicitarea. Un consultant RSistems vă va contacta în cel mai scurt timp pentru demo. "
				f"(ID lead: {lead.id})\n\nDoriți să mai aflați ceva despre sistem?"
			)
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

		# Unknown step -> reset
		_store.update_meta(conversation_id, {"lead": {"active": False, "step": None, "draft": {}}})

	# Stage 1: collect business type
	if stage == "awaiting_business_type":
		bt = _extract_business_type(user_text)
		if not bt:
			reply = "Mulțumesc! Îmi puteți spune tipul locației? (Restaurant / Cafenea / Bar-Pub / Fast-food / Delivery / Lanț de locații)"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

		_store.update_meta(conversation_id, {"business_type": bt, "stage": "chatting"})
		reply = f"Perfect, am notat: {bt}. Ce ați dori să aflați despre sistem?"
		_store.append(conversation_id, {"role": "assistant", "content": reply})
		return jsonify({"conversation_id": conversation_id, "reply": reply})

	# If user is confirming human-contact escalation, start lead capture
	if meta.get("pending_human_contact_confirm"):
		if _detect_yes(user_text):
			_store.update_meta(
				conversation_id,
				{
					"lead": {"active": True, "step": "name", "draft": {}},
					"pending_human_contact_confirm": False,
				},
			)
			reply = "Sigur. Cum vă numiți?"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})
		if _detect_no(user_text):
			_store.update_meta(conversation_id, {"pending_human_contact_confirm": False})
			reply = "Spuneți-mi, vă rog, ce doriți să aflați și încerc să vă ajut cu ce am disponibil."
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})
		reply = "Doar ca să confirm: doriți să vă contactăm? (da/nu)"
		_store.append(conversation_id, {"role": "assistant", "content": reply})
		return jsonify({"conversation_id": conversation_id, "reply": reply})

	# If the assistant just invited the user to a demo, accept a simple "da" as confirmation.
	# This covers cases where the user replies positively without repeating the word "demo".
	history = _store.get(conversation_id)
	last_assistant = _last_assistant_message(history)
	if _assistant_asked_demo(last_assistant):
		label = _classify_demo_intent_hybrid(
			user_text=user_text,
			last_assistant=last_assistant,
			api_key=current_app.config.get("OPENAI_API_KEY", ""),
			model=current_app.config.get("OPENAI_MODEL", "gpt-4o-mini"),
		)
		if label == "WANTS_DEMO":
			_store.update_meta(
				conversation_id,
				{"lead": {"active": True, "step": "name", "draft": {}}, "pending_human_contact_confirm": False},
			)
			reply = "Super! Pentru a vă programa un demo, cum vă numiți?"
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})
		if label == "DOESNT_WANT_DEMO":
			reply = "Spuneți-mi ce ați dori să aflați despre sistem."
			_store.append(conversation_id, {"role": "assistant", "content": reply})
			return jsonify({"conversation_id": conversation_id, "reply": reply})

	# If user asks for demo at any time in chatting stage, start lead capture
	if _classify_demo_intent_hybrid(
		user_text=user_text,
		last_assistant=last_assistant,
		api_key=current_app.config.get("OPENAI_API_KEY", ""),
		model=current_app.config.get("OPENAI_MODEL", "gpt-4o-mini"),
	) == "WANTS_DEMO":
		_store.update_meta(conversation_id, {"lead": {"active": True, "step": "name", "draft": {}}, "pending_human_contact_confirm": False})
		reply = "Super! Pentru a vă programa un demo, cum vă numiți?"
		_store.append(conversation_id, {"role": "assistant", "content": reply})
		return jsonify({"conversation_id": conversation_id, "reply": reply})

	# Retrieve KB snippets for this user message (MVP)
	project_root = os.path.abspath(os.path.join(current_app.root_path, os.pardir))
	kb_dir = os.path.join(project_root, "kb")
	index_path = os.path.join(project_root, "kb_index.json")
	kb = KnowledgeBase(
		kb_dir=kb_dir,
		index_path=index_path,
		openai_api_key=current_app.config.get("OPENAI_API_KEY", ""),
		embedding_model=current_app.config.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
	)
	try:
		snippets = kb.search(user_text)
	except ValueError as exc:
		return (
			jsonify({"error": "config_error", "message": str(exc)}),
			500,
		)

	_store.append(conversation_id, {"role": "user", "content": user_text})
	history = _store.get(conversation_id)

	if not snippets:
		fallback = (
			"Nu sunt sigur că am înțeles exact. "
			"Doriți să vă contactăm? (da/nu)"
		)
		_store.update_meta(conversation_id, {"pending_human_contact_confirm": True})
		_store.append(conversation_id, {"role": "assistant", "content": fallback})
		return jsonify({"conversation_id": conversation_id, "reply": fallback})

	kb_context = "\n\n".join(snippets)
	messages = (
		[
			{
				"role": "system",
				"content": _system_prompt()
				+ "\n\nBază de cunoștințe (RSistems):\n"
				+ kb_context,
			}
		]
		+ [m for m in history if m.get("role") != "system"]
	)

	client = LLMClient(
		api_key=current_app.config.get("OPENAI_API_KEY", ""),
		model=current_app.config.get("OPENAI_MODEL", "gpt-4o-mini"),
	)

	try:
		reply = _strip_leading_padding(client.chat(messages=messages))
	except ValueError as exc:
		return (
			jsonify({"error": "config_error", "message": str(exc)}),
			500,
		)

	_store.append(conversation_id, {"role": "assistant", "content": reply})
	return jsonify({"conversation_id": conversation_id, "reply": reply})

