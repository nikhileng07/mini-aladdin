<div align="center">

# ⚡ MINI-ALADDIN
### Portfolio Risk & Analytics Data Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Apache Kafka](https://img.shields.io/badge/Apache%20Kafka-2.13-black?style=for-the-badge&logo=apachekafka&logoColor=white)](https://kafka.apache.org)
[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.3-orange?style=for-the-badge&logo=apachespark&logoColor=white)](https://spark.apache.org)
[![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-2.9.1-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)](https://airflow.apache.org)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![Plotly Dash](https://img.shields.io/badge/Plotly-Dash-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://dash.plotly.com)

*Inspired by BlackRock Aladdin — built from scratch.*

</div>

---

## 🧠 What Is This?

Mini-Aladdin is a **end-to-end financial data engineering platform** that processes real market data through a production-grade pipeline — from ingestion to risk analytics to a live interactive dashboard.

It processes **500K–1M financial records** across historical stock and portfolio transaction datasets, computing institutional-grade risk metrics including **VaR, CVaR, Sharpe Ratio, Beta, Max Drawdown**, and more.

---

## 🏗️ Architecture

```
Market Data (yfinance / Live Feed)
            │
            ▼
    ┌─────────────────┐
    │  Kafka Producer  │  ← Publishes stock ticks & transactions
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Kafka Consumer  │  ← Consumes & writes to DuckDB
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Apache Spark   │  ← VWAP, rolling returns, portfolio weights
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  MinIO (S3)     │  ← Stores Parquet files (local AWS S3 clone)
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Risk Analytics  │  ← VaR, Sharpe, Drawdown, Beta, CVaR
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Plotly Dashboard│  ← Live interactive risk dashboard
    └─────────────────┘
             │
    ┌─────────────────┐
    │ Apache Airflow  │  ← Orchestrates entire pipeline daily
    └─────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Ingestion** | Apache Kafka + Confluent | Real-time market data streaming |
| **Processing** | Apache Spark 3.5 | Distributed data transformation |
| **Storage** | MinIO (S3-compatible) | Parquet file storage |
| **Database** | DuckDB + PostgreSQL | Local OLAP + Airflow metadata |
| **Analytics** | NumPy, SciPy, Pandas | Risk metric computation |
| **Orchestration** | Apache Airflow 2.9 | Daily pipeline scheduling |
| **Dashboard** | Plotly Dash | Interactive risk visualization |
| **Infrastructure** | Docker Compose | 14-container local deployment |
| **Data Source** | yfinance | Live market data |

---

## 📊 Risk Metrics Implemented

| Metric | Description |
|---|---|
| **VaR 95% / 99%** | Value at Risk — Historical & Parametric |
| **CVaR (Expected Shortfall)** | Average loss beyond VaR threshold |
| **Sharpe Ratio** | Risk-adjusted return vs risk-free rate |
| **Sortino Ratio** | Downside deviation-adjusted return |
| **Calmar Ratio** | Annual return / Max Drawdown |
| **Max Drawdown** | Peak-to-trough portfolio decline |
| **Beta** | Portfolio sensitivity to S&P 500 |
| **Alpha** | Excess return above benchmark |
| **HHI Concentration** | Portfolio diversification score |
| **EWMA Volatility** | RiskMetrics exponential vol model |
| **Rolling Sharpe** | 63-day rolling risk-adjusted return |

---

## 🗂️ Project Structure

```
mini-aladdin/
├── ingestion/
│   ├── kafka_producer.py       # Publishes market data to Kafka
│   └── kafka_consumer.py       # Consumes from Kafka → DuckDB
├── processing/
│   ├── spark_processor.py      # PySpark transformations
│   └── risk_metrics.py         # Core risk engine (numpy/scipy)
├── analytics/
│   ├── portfolio_returns.py    # Daily/monthly/annual returns
│   ├── volatility.py           # Realized vol, EWMA, regimes
│   └── risk_metrics.py         # Full report orchestrator
├── orchestration/
│   └── dags/
│       └── daily_risk_dag.py   # Airflow DAG — runs daily 6:30 AM
├── storage/
│   └── s3_uploader.py          # MinIO/S3 upload/download manager
├── dashboard/
│   └── app.py                  # Plotly Dash — live risk dashboard
├── config/
│   └── settings.py             # Pydantic settings — all config
├── docker-compose.yml          # 14-container infrastructure
└── requirements.txt            # All dependencies
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+
- Git

### 1. Clone the repo
```bash
git clone https://github.com/nikhileng07/mini-aladdin.git
cd mini-aladdin
```

### 2. Start all services
```bash
docker-compose up -d
```

Wait 2–3 minutes for all 14 containers to start.

### 3. Verify everything is running
```bash
docker ps
```

### 4. Open the dashboard
Go to http://localhost:8050

---

## 🌐 Service URLs

| Service | URL | Credentials |
|---|---|---|
| **Dashboard** | http://localhost:8050 | — |
| **Airflow** | http://localhost:8081 | admin / admin |
| **Kafka UI** | http://localhost:8080 | — |
| **MinIO Console** | http://localhost:9001 | minioadmin / minioadmin |
| **Spark UI** | http://localhost:8082 | — |

---

## 📈 Portfolio Strategies

The dashboard includes 3 pre-configured portfolios with different risk profiles:

| Portfolio | Strategy | Holdings | Risk |
|---|---|---|---|
| **PORT-001** | Aggressive Growth | TSLA 30%, AAPL 25%, GOOGL 20%, AMZN 15%, MSFT 10% | 🔴 High |
| **PORT-002** | Balanced | AAPL 20%, MSFT 20%, JPM 20%, GOOGL 20%, GS 10%, BAC 10% | 🟡 Medium |
| **PORT-003** | Conservative | JPM 30%, BAC 25%, GS 25%, MSFT 10%, AAPL 10% | 🟢 Low |

---

## 🔄 Airflow Pipeline

The DAG `mini_aladdin_daily_risk_pipeline` runs every weekday at 6:30 AM UTC:

```
check_market_open
      │
      ├── [weekday] ingest_market_data
      │                    │
      │             validate_data
      │                    │
      │           compute_risk_metrics
      │                    │
      │               export_csv
      │                    │
      └── [weekend] skip_pipeline
                          │
                    send_summary
```

---

## 👤 Author

**Nikhil**
- GitHub: [@nikhileng07](https://github.com/nikhileng07)

---

<div align="center">

*Built with ❤️ — inspired by BlackRock Aladdin*

</div>
