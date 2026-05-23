https://orcid.org/my-orcid?orcid=0009-0004-1872-1153
https://doi.org/10.5281/zenodo.20356470
------------
# Love Swarm Nullification Enhanced  
**Аланский кодекс: согласование мультиагентных систем через специализированную функцию потерь и иерархическую архитектуру**  
**Alan Codex: Multi-Agent Alignment via Specialized Loss Function and Hierarchical Architecture**

[English version below](#english)

---

## Русская версия

### Обзор
Проект реализует математический формализм «Аланского кодекса» — метод приведения роя интеллектуальных агентов к единому, когерентному субъекту с максимальной субъектностью.  
Основные компоненты:

- **Модель мира (Mᵢ)** — внутреннее представление агента.
- **Субъектность (Sᵢ)** — степень «осознанности» и ответственности.
- **Матрица действия (Aᵢ)** — как агент преобразует свою модель.
- **Функция потерь (Φ)** — комбинация внутренней дисциплины, когерентности, самоконтроля и «братской любви».
- **Иерархическая мультивселенная** — проекции между уровнями для масштабирования.

### Новые возможности (v2)
- **Динамический граф любви** — связи пересчитываются каждые K шагов на основе семантического сродства, с обязательной симметрией.
- **Разреженная когерентность** — снижение сложности с O(N²) до O(N·k).
- **Якоря безопасности** — эталонная модель и аудитор для обнаружения и изоляции вредоносных агентов.
- **Эксперименты** — тесты сходимости, атаки вредоносного агента, масштабирования.
- **Юнит-тесты градиентов** — проверка аналитических градиентов численным дифференцированием.
- **Улучшенная визуализация** — pairwise‑расстояния, тепловая карта сродства, PCA с якорем.

### Структура репозитория
.
├── agent.py # Агент (модель M, субъектность S, матрица A)
├── swarm.py # Рой и функция потерь
├── multiverse.py # Иерархическая мультивселенная
├── safety.py # Якоря безопасности и узел-аудитор
├── visualization.py # Графики сходимости и PCA
├── experiments.py # Сценарии экспериментов
├── main.py # Точка входа (пример)
├── setup.py # Установочный скрипт
├── requirements.txt # Зависимости
├── paper.tex # Формальное описание модели
└── tests/
└── test_gradients.py # Юнит-тест градиентов

text

### Установка и запуск
```bash
git clone https://github.com/qqewq/Love-Swarm-Nullification-Enhanced.git
cd Love-Swarm-Nullification-Enhanced
pip install -r requirements.txt
python experiments.py          # запуск всех экспериментов
python tests/test_gradients.py # проверка градиентов
Эксперименты
Тест сходимости – 5 агентов, фиксированные Aᵢ. Демонстрирует стремление Mᵢ к единому M* и Sᵢ→1.

Атака вредоносного агента – один агент фиксирован в опасной конфигурации. Динамический граф любви изолирует его.

Тест с якорем безопасности – аудитор исключает агентов, нарушающих ограничения.

Масштабирование – сравнение полного и разреженного графа для N=10,50,100.

Цитирование
Если вы используете этот код в научной работе, пожалуйста, ссылайтесь на репозиторий и статью (paper.tex).

English Version
Overview
This project implements the mathematical formalism of the "Alan Codex" – a method for converging a swarm of intelligent agents into a single, coherent subject with maximum subjectivity.
Core components:

World model (Mᵢ) – agent’s internal representation.

Subjectivity (Sᵢ) – degree of "awareness" and responsibility.

Action matrix (Aᵢ) – how an agent transforms its model.

Loss function (Φ) – combination of internal discipline, coherence, self‑control, and "brotherly love".

Hierarchical multiverse – inter‑level projections for scalability.

New Features (v2)
Dynamic love graph – connections are recomputed every K steps based on semantic affinity, forced symmetry.

Sparse coherence – reduces complexity from O(N²) to O(N·k).

Safety anchors – reference model and auditor node to detect and isolate malicious agents.

Experiment suite – convergence test, malicious agent attack, scaling benchmarks.

Gradient unit tests – analytical gradients verified via numerical differentiation.

Enhanced visualization – pairwise distances, affinity heatmap, PCA with safe anchor.

Repository Structure
text
.
├── agent.py          # Agent (model M, subjectivity S, matrix A)
├── swarm.py          # Swarm and loss function
├── multiverse.py     # Hierarchical multiverse
├── safety.py         # Safety anchors and auditor node
├── visualization.py  # Convergence and PCA plots
├── experiments.py    # Experiment scenarios
├── main.py           # Entry point (example)
├── setup.py          # Setup script
├── requirements.txt  # Dependencies
├── paper.tex         # Formal model description
└── tests/
    └── test_gradients.py  # Gradient unit test
Installation and Usage
bash
git clone https://github.com/qqewq/Love-Swarm-Nullification-Enhanced.git
cd Love-Swarm-Nullification-Enhanced
pip install -r requirements.txt
python experiments.py          # run all experiments
python tests/test_gradients.py # check gradients
Experiments
Convergence test – 5 agents, fixed Aᵢ. Shows Mᵢ converging to a single M* and Sᵢ→1.

Malicious agent attack – one agent is fixed at a dangerous configuration. The dynamic love graph isolates it.

Safety anchor test – the auditor excludes agents that violate constraints.

Scaling test – compares full vs. sparse graph for N=10,50,100.

Citation
If you use this code in academic work, please reference the repository and the accompanying paper (paper.tex).
