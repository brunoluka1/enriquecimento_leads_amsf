import os
import json
import logging
from contextlib import asynccontextmanager

import httpx
import anthropic
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

APOLLO_API_KEY = os.getenv("APOLLO_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PIPEDRIVE_API_TOKEN = os.getenv("PIPEDRIVE_API_TOKEN")


# ── Modelos ──────────────────────────────────────────────────────────────────

class EnrichRequest(BaseModel):
    email: str | None = None
    phone: str | None = None


# ── Apollo ───────────────────────────────────────────────────────────────────

async def apollo_match_person(client: httpx.AsyncClient, email: str | None = None, phone: str | None = None) -> dict | None:
    """Busca pessoa no Apollo pelo email ou telefone."""
    payload = {"reveal_personal_emails": True}
    if email:
        payload["email"] = email
    if phone:
        payload["phone_number"] = phone
    resp = await client.post(
        "https://api.apollo.io/v1/people/match",
        headers={"Content-Type": "application/json", "x-api-key": APOLLO_API_KEY},
        json=payload,
    )
    if resp.status_code != 200:
        logger.warning("Apollo person match falhou: %s", resp.text)
        return None
    return resp.json().get("person")


async def apollo_enrich_org(client: httpx.AsyncClient, domain: str) -> dict | None:
    """Enriquece organização pelo domínio."""
    resp = await client.get(
        "https://api.apollo.io/v1/organizations/enrich",
        headers={"x-api-key": APOLLO_API_KEY},
        params={"domain": domain},
    )
    if resp.status_code != 200:
        logger.warning("Apollo org enrich falhou: %s", resp.text)
        return None
    return resp.json().get("organization")


async def apollo_search_people(client: httpx.AsyncClient, org_id: str) -> list:
    """Busca funcionários da organização."""
    resp = await client.post(
        "https://api.apollo.io/v1/mixed_people/search",
        headers={"Content-Type": "application/json", "x-api-key": APOLLO_API_KEY},
        json={"organization_ids": [org_id], "per_page": 100, "page": 1},
    )
    if resp.status_code != 200:
        logger.warning("Apollo people search falhou: %s", resp.text)
        return []
    return resp.json().get("people", [])


# ── Firecrawl ────────────────────────────────────────────────────────────────

async def firecrawl_scrape(client: httpx.AsyncClient, url: str) -> str | None:
    """Faz scraping do site da empresa via Firecrawl."""
    resp = await client.post(
        "https://api.firecrawl.dev/v1/scrape",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        },
        json={
            "url": url,
            "formats": ["markdown"],
            "onlyMainContent": True,
            "blockAds": True,
            "removeBase64Images": True,
            "timeout": 30000,
        },
        timeout=60,
    )
    if resp.status_code != 200:
        logger.warning("Firecrawl scrape falhou: %s", resp.text)
        return None
    data = resp.json().get("data", {})
    return data.get("markdown")


# ── Perplexity ───────────────────────────────────────────────────────────────

async def perplexity_research(client: httpx.AsyncClient, company_name: str, industry: str) -> str | None:
    """Pesquisa de mercado sobre empresa e segmento via Perplexity."""
    prompt = f"""Pesquise sobre a empresa "{company_name}" e o segmento "{industry}".

Retorne APENAS um JSON válido com a seguinte estrutura:

{{
  "empresa": {{
    "nome": "",
    "resumo": "",
    "noticias_recentes": [
      {{
        "titulo": "",
        "resumo": "",
        "data": "",
        "fonte": "",
        "url": ""
      }}
    ],
    "posicionamento_mercado": "",
    "destaques": []
  }},
  "segmento": {{
    "nome": "",
    "tendencias_atuais": [],
    "principais_players": [],
    "analise_geral": "",
    "oportunidades": [],
    "ameacas": []
  }},
  "data_pesquisa": ""
}}

Foque em informações dos últimos 90 dias. Priorize fontes jornalísticas, relatórios setoriais e press releases oficiais."""

    resp = await client.post(
        "https://api.perplexity.ai/chat/completions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        },
        json={
            "model": "sonar-deep-research",
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um analista de inteligência de mercado especializado em pesquisa B2B. Sua tarefa é retornar sempre em JSON estruturado, sem markdown, sem explicações fora do JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 4000,
            "temperature": 0.7,
        },
        timeout=120,
    )
    if resp.status_code != 200:
        logger.warning("Perplexity research falhou: %s", resp.text)
        return None
    choices = resp.json().get("choices", [])
    if not choices:
        return None
    return choices[0].get("message", {}).get("content")


# ── Claude (Anthropic) ──────────────────────────────────────────────────────

async def claude_generate_summary(
    person: dict,
    organization: dict | None,
    employees: list,
    website_content: str | None,
    market_research: str | None,
) -> str | None:
    """Gera resumo de enriquecimento do lead usando Claude."""
    employees_text = ""
    for i, emp in enumerate(employees[:20], 1):
        name = emp.get("name", "N/A")
        title = emp.get("title", "N/A")
        employees_text += f"{i}. {name} — {title}\n"

    prompt = f"""You are a B2B sales intelligence assistant. Analyze the following lead data from Apollo and the company website content scraped by Firecrawl, then generate a concise lead enrichment summary for the CRM.

## Lead Data from Apollo:
{person.get('name', 'N/A')}
{person.get('linkedin_url', 'N/A')}
{person.get('title', 'N/A')}
{person.get('headline', 'N/A')}
{person.get('city', '')} {person.get('state', '')}
{person.get('email', 'N/A')}
{person.get('seniority', 'N/A')}

## Company Information
{organization.get('name', 'N/A') if organization else 'N/A'}
{organization.get('website_url', 'N/A') if organization else 'N/A'}
{organization.get('city', 'N/A') if organization else 'N/A'}
{organization.get('state', 'N/A') if organization else 'N/A'}
Total employees found: {len(employees)}

## Company Website Content (scraped by Firecrawl):
{(website_content or 'Não disponível')[:5000]}

## Employees Information:
{employees_text or 'Não disponível'}

## Market Research (Perplexity):
{market_research or 'Não disponível'}

Toda a resposta em português."""

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    try:
        message = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        logger.error("Claude falhou: %s", e)
        return None


# ── Pipedrive ────────────────────────────────────────────────────────────────

PIPEDRIVE_BASE = "https://api.pipedrive.com/v1"


async def pipedrive_find_person(client: httpx.AsyncClient, email: str | None = None, phone: str | None = None) -> int | None:
    """Busca pessoa no Pipedrive pelo email ou telefone."""
    term = email or phone
    field = "email" if email else "phone"
    resp = await client.get(
        f"{PIPEDRIVE_BASE}/persons/search",
        params={
            "term": term,
            "fields": field,
            "exact_match": "true",
            "limit": 1,
            "api_token": PIPEDRIVE_API_TOKEN,
        },
    )
    if resp.status_code != 200:
        logger.warning("Pipedrive search falhou: %s", resp.text)
        return None
    data = resp.json().get("data", {})
    items = data.get("items", [])
    if not items:
        logger.warning("Pessoa não encontrada no Pipedrive: %s", term)
        return None
    return items[0].get("item", {}).get("id")


async def pipedrive_create_note(client: httpx.AsyncClient, person_id: int, content: str) -> bool:
    """Cria nota no Pipedrive vinculada à pessoa."""
    resp = await client.post(
        f"{PIPEDRIVE_BASE}/notes",
        params={"api_token": PIPEDRIVE_API_TOKEN},
        json={"content": content, "person_id": person_id},
    )
    if resp.status_code not in (200, 201):
        logger.warning("Pipedrive create note falhou: %s", resp.text)
        return False
    logger.info("Nota criada no Pipedrive para person_id=%s", person_id)
    return True


# ── Fluxo Principal ─────────────────────────────────────────────────────────

async def enrich_lead(email: str | None = None, phone: str | None = None) -> dict:
    """Executa o fluxo completo de enriquecimento."""
    identifier = email or phone
    logger.info("Iniciando enriquecimento para: %s", identifier)
    result = {"email": email, "phone": phone, "status": "processing", "steps": {}}

    async with httpx.AsyncClient(timeout=60) as client:
        # 1. Apollo — buscar pessoa
        person = await apollo_match_person(client, email=email, phone=phone)
        if not person:
            result["status"] = "error"
            result["error"] = "Pessoa não encontrada no Apollo"
            logger.error("Pessoa não encontrada no Apollo: %s", identifier)
            return result
        result["steps"]["apollo_person"] = "ok"
        logger.info("Apollo person match: %s", person.get("name"))

        # 2. Apollo — enriquecer organização
        org_domain = person.get("organization", {}).get("primary_domain")
        organization = None
        if org_domain:
            organization = await apollo_enrich_org(client, org_domain)
            result["steps"]["apollo_org"] = "ok" if organization else "skipped"
        else:
            result["steps"]["apollo_org"] = "skipped"

        # 3. Apollo — buscar funcionários
        org_id = None
        if organization:
            org_id = organization.get("id")
        elif person.get("organization", {}).get("id"):
            org_id = person["organization"]["id"]

        employees = []
        if org_id:
            employees = await apollo_search_people(client, org_id)
            result["steps"]["apollo_employees"] = f"found {len(employees)}"
        else:
            result["steps"]["apollo_employees"] = "skipped"

        # 4. Firecrawl — scraping do site
        website_content = None
        website_url = organization.get("website_url") if organization else None
        if website_url:
            website_content = await firecrawl_scrape(client, website_url)
            result["steps"]["firecrawl"] = "ok" if website_content else "failed"
        else:
            result["steps"]["firecrawl"] = "skipped"

        # 5. Perplexity — pesquisa de mercado
        market_research = None
        company_name = organization.get("name") if organization else person.get("organization", {}).get("name")
        industry = organization.get("industry") if organization else ""
        if company_name:
            market_research = await perplexity_research(client, company_name, industry or "")
            result["steps"]["perplexity"] = "ok" if market_research else "failed"
        else:
            result["steps"]["perplexity"] = "skipped"

        # 6. Claude — gerar resumo
        summary = await claude_generate_summary(person, organization, employees, website_content, market_research)
        if not summary:
            result["status"] = "partial"
            result["error"] = "Claude não gerou resumo"
            return result
        result["steps"]["claude"] = "ok"

        # 7. Pipedrive — buscar pessoa e criar nota
        person_id = await pipedrive_find_person(client, email=email, phone=phone)
        if person_id:
            note_created = await pipedrive_create_note(client, person_id, summary)
            result["steps"]["pipedrive"] = "ok" if note_created else "failed"
        else:
            result["steps"]["pipedrive"] = "person_not_found"

    result["status"] = "completed"
    result["summary"] = summary
    logger.info("Enriquecimento concluído para: %s", email)
    return result


# ── FastAPI ──────────────────────────────────────────────────────────────────

app = FastAPI(title="Lead Enrichment API", version="1.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/enrich")
async def enrich_manual(req: EnrichRequest):
    """Endpoint manual — envia email ou telefone e recebe resultado."""
    if not req.email and not req.phone:
        return {"status": "error", "error": "Informe email ou phone"}
    result = await enrich_lead(email=req.email, phone=req.phone)
    return result


@app.post("/webhook/pipedrive")
async def webhook_pipedrive(request: Request, background_tasks: BackgroundTasks):
    """Webhook do Pipedrive — dispara enriquecimento em background quando pessoa é criada/atualizada."""
    body = await request.json()

    # Pipedrive envia webhooks com estrutura: { "current": { ... }, "event": "..." }
    current = body.get("current", {})

    # Extrair email e/ou telefone do payload do Pipedrive
    email = None
    phone = None

    email_data = current.get("email", [])
    if isinstance(email_data, list) and email_data:
        email = email_data[0].get("value")
    elif isinstance(email_data, str):
        email = email_data

    phone_data = current.get("phone", [])
    if isinstance(phone_data, list) and phone_data:
        phone = phone_data[0].get("value")
    elif isinstance(phone_data, str):
        phone = phone_data

    if not email and not phone:
        return {"status": "skipped", "reason": "no email or phone found in payload"}

    background_tasks.add_task(enrich_lead, email=email, phone=phone)
    return {"status": "accepted", "email": email, "phone": phone}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
