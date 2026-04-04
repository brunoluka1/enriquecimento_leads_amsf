import os
import json
import asyncio
import logging

import httpx
import anthropic
from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PIPEDRIVE_API_TOKEN = os.getenv("PIPEDRIVE_API_TOKEN")

APIFY_BASE = "https://api.apify.com/v2"


# ── Modelos ──────────────────────────────────────────────────────────────────

class EnrichRequest(BaseModel):
    email: str | None = None
    phone: str | None = None
    name: str | None = None
    company: str | None = None


# ── Apify ────────────────────────────────────────────────────────────────────

async def apify_run_actor(client: httpx.AsyncClient, actor_id: str, run_input: dict) -> list | None:
    """Executa um Actor na Apify e aguarda o resultado."""
    # Iniciar o actor
    resp = await client.post(
        f"{APIFY_BASE}/acts/{actor_id}/runs",
        headers={"Authorization": f"Bearer {APIFY_API_TOKEN}", "Content-Type": "application/json"},
        json=run_input,
        timeout=30,
    )
    logger.info("Apify start actor=%s status=%s", actor_id, resp.status_code)
    if resp.status_code not in (200, 201):
        logger.warning("Apify start falhou: %s", resp.text[:500])
        return None

    run_data = resp.json().get("data", {})
    run_id = run_data.get("id")
    if not run_id:
        return None

    # Aguardar conclusão (polling)
    for _ in range(60):
        await asyncio.sleep(5)
        status_resp = await client.get(
            f"{APIFY_BASE}/actor-runs/{run_id}",
            headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"},
            timeout=15,
        )
        if status_resp.status_code != 200:
            continue
        status = status_resp.json().get("data", {}).get("status")
        logger.info("Apify run=%s status=%s", run_id, status)
        if status == "SUCCEEDED":
            break
        if status in ("FAILED", "ABORTED", "TIMED-OUT"):
            logger.warning("Apify run falhou com status: %s", status)
            return None
    else:
        logger.warning("Apify run timeout após 5 minutos")
        return None

    # Buscar resultados do dataset
    dataset_id = status_resp.json().get("data", {}).get("defaultDatasetId")
    if not dataset_id:
        return None

    items_resp = await client.get(
        f"{APIFY_BASE}/datasets/{dataset_id}/items",
        headers={"Authorization": f"Bearer {APIFY_API_TOKEN}"},
        timeout=30,
    )
    if items_resp.status_code != 200:
        return None
    return items_resp.json()


async def apify_google_search(client: httpx.AsyncClient, query: str) -> str | None:
    """Busca no Google via Apify para encontrar o LinkedIn da pessoa."""
    results = await apify_run_actor(client, "apify~google-search-scraper", {
        "queries": query,
        "maxPagesPerQuery": 1,
        "resultsPerPage": 5,
        "languageCode": "pt",
    })
    if not results:
        return None

    # Procurar URL do LinkedIn nos resultados
    for result in results:
        organic = result.get("organicResults", [])
        for item in organic:
            url = item.get("url", "")
            if "linkedin.com/in/" in url:
                logger.info("LinkedIn encontrado via Google: %s", url)
                return url
    return None


async def apify_linkedin_profile(client: httpx.AsyncClient, linkedin_url: str) -> dict | None:
    """Scrape do perfil LinkedIn via Apify."""
    results = await apify_run_actor(client, "anchor~linkedin-profile-scraper", {
        "profileUrls": [linkedin_url],
    })
    if not results or len(results) == 0:
        return None
    logger.info("LinkedIn profile scraped: %s", results[0].get("fullName", "N/A"))
    return results[0]


async def apify_linkedin_company(client: httpx.AsyncClient, company_url: str) -> dict | None:
    """Scrape da página da empresa no LinkedIn via Apify."""
    results = await apify_run_actor(client, "anchor~linkedin-company-scraper", {
        "companyUrls": [company_url],
    })
    if not results or len(results) == 0:
        return None
    logger.info("LinkedIn company scraped: %s", results[0].get("name", "N/A"))
    return results[0]


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

    # Dados da pessoa (do LinkedIn via Apify)
    person_text = "Não disponível"
    if person:
        person_text = f"""Nome: {person.get('fullName', 'N/A')}
LinkedIn: {person.get('linkedInUrl', 'N/A')}
Cargo: {person.get('title', 'N/A')}
Headline: {person.get('headline', 'N/A')}
Localização: {person.get('location', 'N/A')}
Resumo: {person.get('summary', 'N/A')}"""

    # Dados da empresa (do LinkedIn via Apify)
    company_text = "Não disponível"
    if company:
        company_text = f"""Nome: {company.get('name', 'N/A')}
Website: {company.get('website', 'N/A')}
Setor: {company.get('industry', 'N/A')}
Tamanho: {company.get('employeeCount', 'N/A')} funcionários
Localização: {company.get('headquarters', 'N/A')}
Descrição: {company.get('description', 'N/A')[:1000]}"""

        # Funcionários-chave
        employees = company.get("employees", [])
        if employees:
            company_text += "\n\nFuncionários-chave:"
            for i, emp in enumerate(employees[:20], 1):
                company_text += f"\n{i}. {emp.get('name', 'N/A')} — {emp.get('title', 'N/A')}"

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

        # 1. Pipedrive — buscar pessoa para pegar nome e empresa
        person_id, pipedrive_person = await pipedrive_find_person(client, email=email, phone=phone)
        if not person_id:
            result["status"] = "error"
            result["error"] = "Pessoa não encontrada no Pipedrive"
            return result
        result["steps"]["pipedrive_search"] = "ok"

        person_name = pipedrive_person.get("name", "")
        org_name = pipedrive_person.get("organization", {}).get("name", "") if pipedrive_person.get("organization") else ""
        logger.info("Pipedrive: %s (%s)", person_name, org_name)

        # 2. Apify Google Search — encontrar LinkedIn da pessoa
        search_query = f"{person_name} {org_name} site:linkedin.com/in"
        linkedin_url = await apify_google_search(client, search_query)
        result["steps"]["google_search"] = "ok" if linkedin_url else "not_found"

        # 3. Apify LinkedIn Profile — dados da pessoa
        linkedin_person = None
        if linkedin_url:
            linkedin_person = await apify_linkedin_profile(client, linkedin_url)
            result["steps"]["linkedin_profile"] = "ok" if linkedin_person else "failed"
        else:
            result["steps"]["linkedin_profile"] = "skipped"

        # 4. Apify LinkedIn Company — dados da empresa
        linkedin_company = None
        company_linkedin_url = None
        if linkedin_person:
            company_linkedin_url = linkedin_person.get("companyUrl") or linkedin_person.get("companyLinkedInUrl")
        if company_linkedin_url:
            linkedin_company = await apify_linkedin_company(client, company_linkedin_url)
            result["steps"]["linkedin_company"] = "ok" if linkedin_company else "failed"
        else:
            result["steps"]["linkedin_company"] = "skipped"

        # 5. Firecrawl — scraping do site da empresa
        website_content = None
        website_url = None
        if linkedin_company:
            website_url = linkedin_company.get("website")
        if website_url:
            website_content = await firecrawl_scrape(client, website_url)
            result["steps"]["firecrawl"] = "ok" if website_content else "failed"
        else:
            result["steps"]["firecrawl"] = "skipped"

        # 6. Perplexity — pesquisa de mercado
        market_research = None
        company_name = org_name or (linkedin_company.get("name") if linkedin_company else None)
        industry = linkedin_company.get("industry", "") if linkedin_company else ""
        if company_name:
            market_research = await perplexity_research(client, company_name, industry)
            result["steps"]["perplexity"] = "ok" if market_research else "failed"
        else:
            result["steps"]["perplexity"] = "skipped"

        # 7. Claude — gerar resumo
        summary = await claude_generate_summary(linkedin_person, linkedin_company, website_content, market_research)
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
