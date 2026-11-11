# ☀️ SolarApp — análise do recurso solar

Aplicativo web (Streamlit) para analisar o recurso solar de qualquer localidade, voltado a aulas e
pesquisa em termofluidos e sistemas termossolares. Interface em português do Brasil; código e
docstrings em inglês; unidades SI explícitas (kWh/m²/dia, MJ/m²/dia, W/m², graus).

## Como executar

```powershell
cd analise-recurso-solar
.\.venv\Scripts\python -m pip install -r requirements-dev.txt   # só na primeira vez
.\.venv\Scripts\python -m streamlit run src/solarapp/app.py
```

O navegador abre em http://localhost:8501. A primeira abertura é lenta (~20 s) porque o `.venv`
está no Google Drive (veja "Aviso" abaixo).

Testes e lint:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check src tests
```

## O que o app faz

- **Entrada:** busca de cidade (Open-Meteo Geocoding), latitude/longitude ou clique no mapa;
  período em anos; inclinação (horizontal, definida ou ótima anual) e azimute; albedo.
- **Dados:** NASA POWER (mensal, diário e horário; GHI, difusa, DNI, Kt, céu claro, TOA, T2M, vento)
  ou arquivo CSV/Excel (Atlas INPE/LABREN ou série própria, com mapeamento de colunas e unidade).
  Cache em disco (`~/.solarapp_cache`, 30 dias), retry com backoff, timeout e `-999` → NaN.
- **Física** (`src/solarapp/physics`, funções puras, com equação e referência a Duffie & Beckman):
  declinação, equação do tempo, hora solar, ângulo horário do pôr do sol, duração do dia, zenital,
  altura e azimute solar; H0 e G0 com excentricidade (Gsc = 1361 W/m²); Kt; fração difusa
  (Erbs, Orgill-Hollands, Collares-Pereira–Rabl); Rb; plano inclinado isotrópico (Liu-Jordan) e
  Hay-Davies; inclinação ótima anual e mensal; carta solar; FV e coletor (indicadores).
- **Abas:** Resumo, Dados mensais, Série diária, Plano inclinado, Trajetória solar, Dados brutos.
- **Exportação:** CSV, Excel e relatório HTML (imprima como PDF pelo navegador).

## Publicar na web e usar no celular (Streamlit Community Cloud)

1. Em https://share.streamlit.io entre com o GitHub e clique em **Create app**.
2. Repositório `analise-recurso-solar`, branch `main`, arquivo principal `src/solarapp/app.py`.
3. Em *Advanced settings* escolha Python 3.12 ou 3.13 e publique (`requirements.txt` é lido sozinho).
4. No Android (Chrome): abra o link → menu ⋮ → **Adicionar à tela inicial** / **Instalar app**.

## Estrutura

```
src/solarapp/
  app.py          interface Streamlit
  analysis.py     tabelas derivadas (cola entre dados e física)
  config.py       constantes
  notes.py        hipóteses e limitações
  data/           nasa_power, geocoding, cache, importers, schema
  physics/        geometry, extraterrestrial, clearness, diffuse, tilted, sunpath, indicators
  plots/          figuras Plotly
  reports/        exportações CSV/Excel/HTML
tests/            pytest (dados com mocks HTTP, física, análise, importação)
```

## Atlas INPE/LABREN

Não há API pública: o Atlas é distribuído como arquivos (CSV/SHP/NetCDF, em **Wh/m²/dia**).
O app só importa arquivos; escolha a unidade correta em "Unidade dos valores".

## Aviso: `.venv` no Google Drive

O projeto está numa pasta sincronizada pelo Google Drive. Isso deixa os imports lentos (pandas ~13 s)
e fez o OpenSSL travar ao ler o `cacert.pem` do certifi. O cliente HTTP usa por isso o
repositório de certificados do sistema (veja `_SystemTrustAdapter`). Se possível, recrie o
`.venv` em disco local para ganhar velocidade.

## Hipóteses e limitações

Veja `src/solarapp/notes.py` (também exibidas no app e no relatório).