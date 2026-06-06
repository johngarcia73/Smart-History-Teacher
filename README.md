# Smart History Teacher: A Personalized Intelligent History Tutor

## Overview

Smart History Teacher is an advanced educational platform designed to provide highly personalized history instruction inside a modern academic environment. At its core, it combines a Retrieval-Augmented Generation (RAG) pipeline with a structured historical ontology and bio-inspired optimization techniques to deliver responses that are accurate, context-aware, and tailored to each learner's profile.

The system uses a SPADE-based multi-agent architecture in which specialized agents collaborate to retrieve, evaluate, enrich, and generate educational content. A search agent combines semantic and lexical retrieval over a knowledge base built with FAISS and BM25, while a crawler agent supplements missing information from the web. A prompt agent adapts the response style to the learning scenario, and profile agents continuously model the student's preferences, behavior, and progress. A dedicated Moodle agent connects the AI engine with the learning management system, allowing the tutor to function as a virtual assistant directly inside Moodle chat.

What makes the project especially distinctive is its personalization layer. Particle Swarm Optimization (PSO) is used to tune generation parameters such as temperature and top-p based on the student's preferred tone and learning style, while Genetic Algorithms help optimize profile clustering and learning-path adaptation. The ontology manager further strengthens the system by linking events, people, periods, and concepts into a verified historical knowledge graph, reducing hallucinations and improving conceptual grounding.

The result is a powerful intelligent tutoring system capable of explaining historical topics, adapting to different teaching styles, and supporting students through a richer and more personalized learning experience.

---

## Key Features

### Multi-Agent Architecture (SPADE)

The platform is built around a distributed network of specialized agents:

* **Search Agent** — Semantic and lexical retrieval using FAISS and BM25.
* **Evaluation Agent** — Filters and ranks retrieved information.
* **Crawler Agent** — Dynamically gathers supplementary information from the web.
* **Prompt Agent** — Manages adaptive prompting strategies and LLM interactions.
* **Profile Agent** — Maintains persistent student profiles and learning preferences.
* **Moodle Agent** — Integrates the tutoring system directly into Moodle.

### Personalized RAG Pipeline

Unlike traditional RAG systems, Smart History Teacher adapts retrieval to the user's profile and educational context.

* Semantic search with Sentence Transformers and FAISS.
* Lexical retrieval using BM25.
* Context-aware retrieval strategies.
* Query expansion through historical ontologies.

### Bio-Inspired Optimization

The platform continuously evolves its behavior to better suit each student.

#### Particle Swarm Optimization (PSO)

Used to dynamically optimize:

* Temperature
* Top-P
* Repetition Penalty
* Creativity level
* Formality level
* Response style

#### Genetic Algorithms

Used for:

* Student clustering
* Learning path optimization
* Communication style adaptation
* Interest modeling

### Ontology-Driven Knowledge Management

The system uses structured historical ontologies to:

* Relate historical events, people, locations, and concepts.
* Infer historical periods and contextual relationships.
* Ground responses in verified knowledge.
* Reduce hallucinations in generated content.

### Moodle Integration

The tutor operates directly within Moodle:

* Monitors student messages.
* Processes educational queries through the multi-agent pipeline.
* Delivers personalized responses back into Moodle chat.
* Functions as a virtual teaching assistant inside existing academic environments.

---

## Architecture

```text
Student
   │
   ▼
Moodle Agent
   │
   ▼
Prompt Agent
   │
   ├── Search Agent (FAISS + BM25)
   │
   ├── Evaluation Agent
   │
   ├── Crawler Agent
   │
   └── Profile Agent
           │
           ▼
PSO + Genetic Optimization
           │
           ▼
Ontology Manager
           │
           ▼
LLM Response Generation
```

---

## Core Technologies

### Artificial Intelligence

* Retrieval-Augmented Generation (RAG)
* Large Language Models (LLMs)
* Adaptive Prompt Engineering

### Information Retrieval

* FAISS
* BM25
* Sentence Transformers

### Multi-Agent Systems

* SPADE
* XMPP-based communication

### Optimization

* Particle Swarm Optimization (PSO)
* Genetic Algorithms

### Knowledge Representation

* RDFLib
* OWL Ontologies
* Turtle (.ttl)

### Integration

* Moodle Web Services API
* Python 3.9+

---

## Example Use Cases

Smart History Teacher can be used in:

* Universities
* Online education platforms
* Research projects
* Intelligent Tutoring Systems (ITS)
* Personalized e-learning environments
* History and humanities courses
* Academic support programs

---

## Educational Benefits

* Personalized explanations adapted to each learner.
* Reduced hallucinations through ontology grounding.
* Dynamic adjustment of teaching style and complexity.
* Integration with existing Moodle infrastructures.
* Continuous adaptation based on student behavior and preferences.

---

## License

This project was developed for educational and research purposes in the field of Intelligent Tutoring Systems (ITS).
