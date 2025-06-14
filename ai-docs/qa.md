# Clarifying Questions:

- Frontend Choice: I suggested FastAPI + Next.js instead of Django for better performance and modern architecture. Are you okay with this, or do you specifically prefer Django?
- Additional Models: The baseline includes XGBoost scores, but it wasn't in your initial model list. Should we include XGBoost as well?
- Data Location: Should the CSV file be included in the project repository, or will it be loaded from an external location?
- Deployment Target: Are you planning to deploy this locally, on a cloud platform, or both?
- Real-time Requirements: Do you need real-time price updates, or is this purely for historical backtesting?
- Performance Constraints: The plan aims for processing within 5 minutes. Is this acceptable, or do you have stricter requirements?

# Clarification Answers

## Clarification Answers

- **Frontend Choice:** Yes, FastAPI + Next.js is fine and preferred over Django for better performance and modern architecture.

- **Additional Models:** We will include XGBoost and all its best variants in our implementation to maximize our chances of beating the baseline metrics.

- **Data Location:** The CSV file will be stored inside the project in a folder named "asset_data". Additionally, there is a folder "baseline" containing the baseline sources (paper and graphs with target metrics to beat in "base_graphs.png").

- **Deployment Target:** Primary development will be on a high-performance desktop with RTX 5090 GPU running WSL2 on Windows 11, but the solution must be compatible with 3-4 year old MacBook Pro hardware to ensure team-wide usability.

- **Real-time Requirements:** Initial MVP will focus on historical backtesting only. The architecture should be designed to be easily extensible for future integration with a real-time data fetching AI agent, but that will be implemented only after APM-AI v1.0 is complete.

- **Performance Constraints:** No strict processing time requirements. The primary goal is to beat the baseline numbers with an elegant, maintainable solution. 

- **Current Project Structure:** 

```bash
.
├── ai-docs
│   ├── implementation.md
│   ├── plan.md
│   ├── prd.md
│   ├── qa.md
│   ├── rules.md
│   └── spec.md
├── asset_data
└── baseline
    ├── Baseline_paper.md
    ├── base_graphs.png
    ├── problem_statement.md
    └── technical_approach.md

4 directories, 10 files
```