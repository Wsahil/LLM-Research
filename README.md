# LLM-Research
**An end‑to‑end set of LLM‑powered tools that answer course questions and drive data‑backed retail decisions at Syracuse University.**

**Overview**  
This repository houses the Data Intelligence Suite I built during my Research Data Scientist assistant appointment It solves two complementary problems:  
Academic Q&A Assistant  
Streamlit + Pinecone + Cohere app that lets students query >250 graduate‑course syllabi in natural language.  
Retail Analytics Pipeline  
Airflow‑orchestrated ETL → MS SQL warehouse → Python/Pandas data‑quality checks → actionable anomaly alerts for campus‑store teams.

**Key Results**  
Q&A Assistant: 93 % of student questions answered in under 10 s → 40 % reduction in faculty email load  
ETL Accuracy: 99 % data‑processing accuracy → eliminated manual extracts  
Cloud & Labor: $5 k / month saved in cloud & labor cost  
Anomaly Detection: $12 k incremental profit per semester through dynamic repricing  

**Features**  
LLM‑Powered Q&A Assistant – A Streamlit chat interface powered by Cohere Generate and a Pinecone vector index (~3 k syllabus chunks). Students submit a natural‑language question and receive a cited answer in under 10 seconds.  
Nightly ETL to MS SQL – Apache Airflow 2 orchestrates ingestion from seven POS, e‑commerce, and inventory feeds, reconciles schemas, and populates a star‑schema warehouse (fact_sales, dim_product, etc.)—no manual extracts required.  
Data‑Quality Pipeline – GitLab CI triggers Pandas + Great Expectations tests post‑ingest; any failing rows are captured in an anomaly_log table for follow‑up, ensuring warehouse reliability.  
Alerting & Reporting – Slack webhooks publish anomaly summaries, and curated tables power Power BI dashboards for inventory, pricing, and executive KPIs.  

**Techstack**  
UI: Streamlit  
LLM / Embeddings: Cohere Generate, Embed‑v3  
Vector Store: Pinecone  
Orchestration: Apache Airflow 2  
Warehouse: MS SQL  
Data Quality: Pandas • Great Expectations  
CI / CD: GitLab  
Hosting: Azure VM • Streamlit Cloud  
