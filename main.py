import os
import json
import logging

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

APOLLO_BASE = "https://api.apollo.io/api/v1"


# ── Modelos ──────────────────────────────────────────────────────────────────

class EnrichRequest(BaseModel):
    email: str | None = None
    phone: str | None = None
    name: str | None = None
    company: str | None = None


# ── Apollo ──────────────────────────────────────────────────────────────────

async def apollo_search_person(client: httpx.AsyncClient, email: str | None = None, phone: str | None = None) -> tuple[dict | None, dict | None]:
    """Busca pessoa via Apollo /contacts/search. Retorna (person, organization)."""
    payload = {"per_page": 1, "page": 1}
    if email:
        payload["q_keywords"] = email
    elif phone:
        payload["q_keywords"] = phone

    resp = await client.post(
        f"{APOLLO_BASE}/contacts/search",
        headers={"Content-Type": "application/json", "X-Api-Key": APOLLO_API_KEY},
        json=payload,
        timeout=30,
    )
    logger.info("Apollo /contacts/search status=%s", resp.status_code)
    logger.info("Apollo response: %s", resp.text[:1000])

    if resp.status_code != 200:
        logger.warning("Apollo falhou: %s", resp.text[:500])
        return None, None

    data = resp.json()
    contacts = data.get("contacts", [])
    if not contacts:
        logger.warning("Apollo não encontrou pessoa para: %s", email or phone)
        return None, None

    person = contacts[0]
    organization = person.get("organization") or person.get("account")

    logger.info("Apollo person: %s — %s @ %s",
                 person.get("name", "N/A"),
                 person.get("title", "N/A"),
                 organization.get("name", "N/A") if organization else "N/A")

    return person, organization


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
    person: dict | None,
    company: dict | None,
    website_content: str | None,
    market_research: str | None,
) -> str | None:
    """Gera resumo de enriquecimento do lead usando Claude."""

    # Dados da pessoa (Apollo)
    person_text = "Não disponível"
    if person:
        person_text = f"""Nome: {person.get('name', 'N/A')}
Email: {person.get('email', 'N/A')}
LinkedIn: {person.get('linkedin_url', 'N/A')}
Cargo: {person.get('title', 'N/A')}
Headline: {person.get('headline', 'N/A')}
Localização: {person.get('city', '')}, {person.get('state', '')}, {person.get('country', '')}
Senioridade: {person.get('seniority', 'N/A')}
Departamento: {', '.join(person.get('departments', [])) if person.get('departments') else 'N/A'}"""

    # Dados da empresa (Apollo)
    company_text = "Não disponível"
    if company:
        company_text = f"""Nome: {company.get('name', 'N/A')}
Website: {company.get('website_url', 'N/A')}
Setor: {company.get('industry', 'N/A')}
Subsetor: {company.get('sub_industry', 'N/A')}
Tamanho: {company.get('estimated_num_employees', 'N/A')} funcionários
Localização: {company.get('city', '')}, {company.get('state', '')}, {company.get('country', '')}
Receita estimada: {company.get('annual_revenue_printed', 'N/A')}
Descrição: {(company.get('short_description') or company.get('description') or 'N/A')[:1000]}
LinkedIn: {company.get('linkedin_url', 'N/A')}
Palavras-chave: {', '.join(company.get('keywords', [])[:10]) if company.get('keywords') else 'N/A'}
Tecnologias: {', '.join(company.get('technology_names', [])[:15]) if company.get('technology_names') else 'N/A'}"""

    prompt = f"""You are a B2B sales intelligence assistant. Analyze the following lead data and generate a concise lead enrichment summary for the CRM.

## Lead Data (Apollo):
{person_text}

## Company Information (Apollo):
{company_text}

## Company Website Content (Firecrawl):
{(website_content or 'Não disponível')[:5000]}

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


async def pipedrive_find_person(client: httpx.AsyncClient, email: str | None = None, phone: str | None = None) -> tuple[int | None, dict | None]:
    """Busca pessoa no Pipedrive pelo email ou telefone. Retorna (id, dados)."""
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
        return None, None
    data = resp.json().get("data", {})
    items = data.get("items", [])
    if not items:
        logger.warning("Pessoa não encontrada no Pipedrive: %s", term)
        return None, None
    item = items[0].get("item", {})
    return item.get("id"), item


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

        # 1. Pipedrive — buscar pessoa
        person_id, pipedrive_person = await pipedrive_find_person(client, email=email, phone=phone)
        if not person_id:
            result["status"] = "error"
            result["error"] = "Pessoa não encontrada no Pipedrive"
            return result
        result["steps"]["pipedrive_search"] = "ok"
        logger.info("Pipedrive: person_id=%s", person_id)

        # 2. Apollo — enriquecer dados da pessoa e empresa
        apollo_person, apollo_org = await apollo_search_person(client, email=email, phone=phone)
        result["steps"]["apollo"] = "ok" if apollo_person else "not_found"

        # 3. Firecrawl — scraping do site da empresa
        website_content = None
        website_url = apollo_org.get("website_url") if apollo_org else None
        if website_url:
            website_content = await firecrawl_scrape(client, website_url)
            result["steps"]["firecrawl"] = "ok" if website_content else "failed"
        else:
            result["steps"]["firecrawl"] = "skipped"

        # 4. Perplexity — pesquisa de mercado
        market_research = None
        company_name = apollo_org.get("name") if apollo_org else None
        industry = apollo_org.get("industry", "") if apollo_org else ""
        if company_name:
            market_research = await perplexity_research(client, company_name, industry)
            result["steps"]["perplexity"] = "ok" if market_research else "failed"
        else:
            result["steps"]["perplexity"] = "skipped"

        # 5. Claude — gerar resumo
        summary = await claude_generate_summary(apollo_person, apollo_org, website_content, market_research)
        if not summary:
            result["status"] = "partial"
            result["error"] = "Claude não gerou resumo"
            return result
        result["steps"]["claude"] = "ok"

        # 6. Pipedrive — criar nota
        note_created = await pipedrive_create_note(client, person_id, summary)
        result["steps"]["pipedrive_note"] = "ok" if note_created else "failed"

    result["status"] = "completed"
    result["summary"] = summary
    logger.info("Enriquecimento concluído para: %s", identifier)
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


class TestRequest(BaseModel):
    company_name: str
    company_website: str | None = None
    industry: str | None = None
    person_name: str | None = None
    person_title: str | None = None


@app.post("/test")
async def test_integrations(req: TestRequest):
    """Endpoint de teste — testa Firecrawl, Perplexity e Claude sem Apollo."""
    result = {"status": "processing", "steps": {}}

    async with httpx.AsyncClient(timeout=60) as client:

        # 1. Firecrawl
        website_content = None
        if req.company_website:
            website_content = await firecrawl_scrape(client, req.company_website)
            result["steps"]["firecrawl"] = "ok" if website_content else "failed"
        else:
            result["steps"]["firecrawl"] = "skipped"

        # 2. Perplexity
        market_research = None
        market_research = await perplexity_research(client, req.company_name, req.industry or "")
        result["steps"]["perplexity"] = "ok" if market_research else "failed"

        # 3. Claude — montar dados fake de pessoa/empresa pra testar
        fake_person = None
        if req.person_name:
            fake_person = {
                "name": req.person_name,
                "title": req.person_title or "N/A",
                "headline": req.person_title or "N/A",
            }

        fake_company = {
            "name": req.company_name,
            "website_url": req.company_website or "N/A",
            "industry": req.industry or "N/A",
        }

        summary = await claude_generate_summary(fake_person, fake_company, website_content, market_research)
        result["steps"]["claude"] = "ok" if summary else "failed"

    result["status"] = "completed"
    result["summary"] = summary
    return result


@app.post("/webhook/pipedrive")
async def webhook_pipedrive(request: Request, background_tasks: BackgroundTasks):
    """Webhook do Pipedrive — dispara enriquecimento em background."""
    body = await request.json()
    current = body.get("current", {})

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
