graph TD
    A[Moodle] --> B[Personality Analyzer]
    B --> C[Profile Manager]
    B --> D[Search Agent]
    D --> E[Evaluation Agent]
    E --> F[Crawler Agent]
    F --> G[Prompt Agent]
    C --> G
    G --> A