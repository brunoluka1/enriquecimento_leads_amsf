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

PROXYCURL_API_KEY = os.getenv("PROXYCURL_API_KEY")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PIPEDRIVE_API_TOKEN = os.getenv("PIPEDRIVE_API_TOKEN")

PROXYCURL_BASE = "https://nubela.co/proxycurl"


# ── Modelos ──────────────────────────────────────────────────────────────────

class EnrichRequest(BaseModel):
    email: str | None = None
    phone: str | None = None
    name: str | None = None
    company: str | None = None


# ── Proxycurl ───────────────────────────────────────────────────────────────

def _proxycurl_headers() -> dict:
    return {"Authorization": f"Bearer {PROXYCURL_API_KEY}"}


async def proxycurl_resolve_email(client: httpx.AsyncClient, email: str) -> str | None:
    """Busca o perfil LinkedIn a partir do email via Proxycurl. Retorna a URL do perfil."""
    resp = await client.get(
        f"{PROXYCURL_BASE}/api/linkedin/profile/resolve/email",
        headers=_proxycurl_headers(),
        params={"email": email, "lookup_depth": "superficial"},
        timeout=30,
    )
    logger.info("Proxycurl resolve email status=%s", resp.status_code)
    if resp.status_code != 200:
        logger.warning("Proxycurl resolve email falhou: %s", resp.text[:500])
        return None

    data = resp.json()
    linkedin_url = data.get("url")
    if linkedin_url:
        logger.info("Proxycurl encontrou LinkedIn: %s", linkedin_url)
    else:
        logger.warning("Proxycurl não encontrou LinkedIn para: %s", email)
    return linkedin_url


async def proxycurl_resolve_phone(client: httpx.AsyncClient, phone: str) -> str | None:
    """Busca o perfil LinkedIn a partir do telefone via Proxycurl."""
    resp = await client.get(
        f"{PROXYCURL_BASE}/api/resolve/phone",
        headers=_proxycurl_headers(),
        params={"phone_number": phone},
        timeout=30,
    )
    logger.info("Proxycurl resolve phone status=%s", resp.status_code)
    if resp.status_code != 200:
        logger.warning("Proxycurl resolve phone falhou: %s", resp.text[:500])
        return None

    data = resp.json()
    linkedin_url = data.get("url")
    if linkedin_url:
        logger.info("Proxycurl encontrou LinkedIn por telefone: %s", linkedin_url)
    return linkedin_url


async def proxycurl_person_profile(client: httpx.AsyncClient, linkedin_url: str) -> dict | None:
    """Busca dados completos do perfil LinkedIn via Proxycurl."""
    resp = await client.get(
        f"{PROXYCURL_BASE}/api/v2/linkedin",
        headers=_proxycurl_headers(),
        params={"linkedin_profile_url": linkedin_url, "skills": "include"},
        timeout=30,
    )
    logger.info("Proxycurl person profile status=%s", resp.status_code)
    if resp.status_code != 200:
        logger.warning("Proxycurl person profile falhou: %s", resp.text[:500])
        return None

    person = resp.json()
    logger.info("Proxycurl person: %s — %s", person.get("full_name", "N/A"), person.get("headline", "N/A"))
    return person


async def proxycurl_company_profile(client: httpx.AsyncClient, linkedin_url: str) -> dict | None:
    """Busca dados da empresa no LinkedIn via Proxycurl."""
    resp = await client.get(
        f"{PROXYCURL_BASE}/api/linkedin/company",
        headers=_proxycurl_headers(),
        params={"url": linkedin_url},
        timeout=30,
    )
    logger.info("Proxycurl company profile status=%s", resp.status_code)
    if resp.status_code != 200:
        logger.warning("Proxycurl company profile falhou: %s", resp.text[:500])
        return None

    company = resp.json()
    logger.info("Proxycurl company: %s — %s", company.get("name", "N/A"), company.get("industry", "N/A"))
    return company


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

    # Dados da pessoa (Proxycurl/LinkedIn)
    person_text = "Não disponível"
    if person:
        # Experiência atual
        experiences = person.get("experiences", [])
        current_exp = ""
        if experiences:
            exp = experiences[0]
            current_exp = f"\nEmpresa atual: {exp.get('company', 'N/A')} — {exp.get('title', 'N/A')}"
            if exp.get("description"):
                current_exp += f"\nDescrição do cargo: {exp['description'][:500]}"

        person_text = f"""Nome: {person.get('full_name', 'N/A')}
LinkedIn: {person.get('public_identifier', 'N/A')}
Headline: {person.get('headline', 'N/A')}
Localização: {person.get('city', '')}, {person.get('state', '')}, {person.get('country_full_name', '')}
Resumo: {(person.get('summary') or 'N/A')[:800]}{current_exp}
Skills: {', '.join(person.get('skills', [])[:15]) if person.get('skills') else 'N/A'}"""

    # Dados da empresa (Proxycurl/LinkedIn)
    company_text = "Não disponível"
    if company:
        company_text = f"""Nome: {company.get('name', 'N/A')}
Website: {company.get('website', 'N/A')}
Setor: {company.get('industry', 'N/A')}
Tamanho: {company.get('company_size_on_linkedin', 'N/A')} funcionários
Localização: {', '.join(filter(None, [company.get('city'), company.get('state'), company.get('country')]))}
Descrição: {(company.get('description') or 'N/A')[:1000]}
LinkedIn: {company.get('linkedin_internal_id', 'N/A')}
Especialidades: {', '.join(company.get('specialities', [])[:10]) if company.get('specialities') else 'N/A'}
Fundada em: {company.get('founded_year', 'N/A')}
Tipo: {company.get('company_type', 'N/A')}"""

    prompt = f"""You are a B2B sales intelligence assistant. Analyze the following lead data and generate a concise lead enrichment summary for the CRM.

## Lead Data (LinkedIn):
{person_text}

## Company Information (LinkedIn):
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

        # 2. Proxycurl — encontrar LinkedIn pelo email ou telefone
        linkedin_url = None
        if email:
            linkedin_url = await proxycurl_resolve_email(client, email)
        if not linkedin_url and phone:
            linkedin_url = await proxycurl_resolve_phone(client, phone)
        result["steps"]["proxycurl_resolve"] = "ok" if linkedin_url else "not_found"

        # 3. Proxycurl — dados do perfil LinkedIn
        person_data = None
        if linkedin_url:
            person_data = await proxycurl_person_profile(client, linkedin_url)
            result["steps"]["proxycurl_person"] = "ok" if person_data else "failed"
        else:
            result["steps"]["proxycurl_person"] = "skipped"

        # 4. Proxycurl — dados da empresa no LinkedIn
        company_data = None
        company_linkedin_url = None
        if person_data:
            experiences = person_data.get("experiences", [])
            if experiences:
                company_linkedin_url = experiences[0].get("company_linkedin_profile_url")
        if company_linkedin_url:
            company_data = await proxycurl_company_profile(client, company_linkedin_url)
            result["steps"]["proxycurl_company"] = "ok" if company_data else "failed"
        else:
            result["steps"]["proxycurl_company"] = "skipped"

        # 5. Firecrawl — scraping do site da empresa
        website_content = None
        website_url = company_data.get("website") if company_data else None
        if website_url:
            website_content = await firecrawl_scrape(client, website_url)
            result["steps"]["firecrawl"] = "ok" if website_content else "failed"
        else:
            result["steps"]["firecrawl"] = "skipped"

        # 6. Perplexity — pesquisa de mercado
        market_research = None
        company_name = company_data.get("name") if company_data else None
        industry = company_data.get("industry", "") if company_data else ""
        if company_name:
            market_research = await perplexity_research(client, company_name, industry)
            result["steps"]["perplexity"] = "ok" if market_research else "failed"
        else:
            result["steps"]["perplexity"] = "skipped"

        # 7. Claude — gerar resumo
        summary = await claude_generate_summary(person_data, company_data, website_content, market_research)
        if not summary:
            result["status"] = "partial"
            result["error"] = "Claude não gerou resumo"
            return result
        result["steps"]["claude"] = "ok"

        # 8. Pipedrive — criar nota
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
