# RSistems Chatbot (Flask) — Skeleton

Flask backend skeleton for an RSistems website chatbot.

## Structure
- `app/` — application package (app factory, config, blueprints)
- `app/api/` — API blueprints (chat, leads, health)
- `app/services/` — business logic (LLM client, knowledge base, lead service)
- `app/models/` — database models
- `kb/` — Markdown knowledge base (RSistems info, FAQ, pricing disclaimer)
- `instance/` — instance-specific runtime files (not committed)

## Quick start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python build_kb_index.py
python init_db.py
python run.py
```

## Test in terminal
1) Set your OpenAI key in `.env` (or set env vars in your shell):
	- `OPENAI_API_KEY=...`
	- `OPENAI_MODEL=gpt-4o-mini`

2) Run the API:
```powershell
python run.py
```

3) In another terminal, run the CLI tester:
```powershell
python cli_chat.py
```

## Conversation flow (current)
- First assistant message asks for business type (Restaurant/Cafenea/Bar/Pub/Fast-food/Delivery/Lanț de locații)
- After business type is provided, you can ask RSistems questions; answers come from `kb/`
- If you ask for a demo (mention "demo"), the bot starts a step-by-step lead capture (name → phone → email → business name → number of locations) and saves to DB
- If a question can't be answered from the KB, the bot asks if you want to be contacted by a human; if you confirm, it starts the same lead capture flow

## Endpoints
- `GET /health` — returns `{ "status": "ok" }`
- `POST /api/chat` — chat endpoint (implemented with OpenAI; greets on first call)
- `POST /api/leads` — creates a lead in the database
- `GET /api/leads` — lists leads (testing)

Example payload:
```json
{
	"name": "Ion Popescu",
	"phone": "+40 7xx xxx xxx",
	"email": "ion@example.com",
	"business_name": "Restaurantul La Ion",
	"type_of_business": "Restaurant",
	"nr_of_locations": 2
}
```

List leads (PowerShell):
```powershell
Invoke-RestMethod http://localhost:5000/api/leads -Method Get
```
