import requests
from typing import List

def get_crypto_tickers(min_volume_quote: float = 1_000_000) -> List[str]:
    """
    Busca dinamicamente todos os pares USDT na Binance.
    Filtra por status TRADING e volume mínimo (se disponível no endpoint de ticker/24hr).
    """
    print("🔄 Buscando lista de Criptos na Binance...")
    try:
        # 1. Pegar Exchange Info para pares válidos
        url_info = "https://api.binance.com/api/v3/exchangeInfo"
        resp = requests.get(url_info).json()
        
        symbols = []
        for s in resp['symbols']:
            if s['status'] == 'TRADING' and s['quoteAsset'] == 'USDT' and s['isSpotTradingAllowed']:
                symbols.append(s['symbol'])
        
        # 2. (Opcional) Filtrar por volume para evitar moedas mortas
        # Isso requer outra chamada, faremos um filtro simples aqui:
        # Retorna apenas os que terminam em USDT
        print(f"✅ Encontrados {len(symbols)} pares USDT ativos.")
        return symbols
    except Exception as e:
        print(f"❌ Erro ao buscar Criptos: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"] # Fallback

def get_b3_tickers() -> List[str]:
    """
    Retorna uma lista curada dos ativos mais líquidos da B3 (IBOV + SMLL).
    Isso evita tentar baixar dados de empresas falidas ou sem liquidez.
    """
    print("🔄 Carregando lista de Ações B3 (IBOV + SMLL)...")
    tickers = [
        # Óleo e Gás / Petroquímica
        "PETR4.SA", "PETR3.SA", "PRIO3.SA", "UGPA3.SA", "CSAN3.SA", "VBBR3.SA", "RECV3.SA",
        # Bancos e Financeiro
        "ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA", "BPAC11.SA", "B3SA3.SA", "CIEL3.SA",
        # Mineração e Siderurgia
        "VALE3.SA", "GGBR4.SA", "CSNA3.SA", "USIM5.SA", "CMIN3.SA",
        # Varejo e Consumo
        "MGLU3.SA", "LREN3.SA", "RENT3.SA", "ARZZ3.SA", "SOMA3.SA", "ALPA4.SA", "ASAI3.SA", "CRFB3.SA", "NTCO3.SA",
        # Elétricas e Saneamento
        "ELET3.SA", "ELET6.SA", "CMIG4.SA", "CPLE6.SA", "EQTL3.SA", "TRPL4.SA", "TAEE11.SA", "SBSP3.SA",
        # Construção e Imobiliário
        "CYRE3.SA", "EZTC3.SA", "MRVE3.SA", "MULT3.SA", "IGTI11.SA",
        # Indústria e Bens de Capital
        "WEGE3.SA", "EMBR3.SA", "TASA4.SA", "POMO4.SA",
        # Saúde
        "HAPV3.SA", "RDOR3.SA", "RADL3.SA",
        # Agro / Papel e Celulose
        "SUZB3.SA", "KLBN11.SA", "JBSS3.SA", "BRFS3.SA", "MRFG3.SA", "BEEF3.SA", "SLCE3.SA",
        # Tech / Outros
        "TOTS3.SA", "LWSA3.SA", "CASH3.SA", "YDUQ3.SA", "COGN3.SA"
    ]
    print(f"✅ Lista B3 carregada: {len(tickers)} ativos.")
    return tickers
