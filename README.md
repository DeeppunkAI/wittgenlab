# WittgenLab 🧪

Un framework integral de evaluación para modelos de IA que proporciona una interfaz unificada para ejecutar métricas, benchmarks y análisis.

## ✨ Características Principales

- **🎯 API Unificada**: Interfaz única para todas las necesidades de evaluación
- **📊 Métricas Integrales**: Métricas n-gram, semánticas, perplexity y personalizadas
- **🏆 Benchmarks Estándar**: GLUE, MMLU, HumanEval, TruthfulQA y más
- **🤖 LLM-as-a-Judge**: Soporte integrado para evaluación basada en LLM
- **👥 Evaluación Humana**: Herramientas para anotación y crowdsourcing
- **📈 Análisis y Visualización**: Análisis de correlación, detección de sesgos y reportes
- **⚡ Fácil de Usar**: API simple e intuitiva con configuración mínima
- **🔧 Extensible**: Sistema de plugins para métricas y benchmarks personalizados

## 🚀 Inicio Rápido

### Instalación

```bash
# Instalación básica
pip install wittgenlab

# Con todas las dependencias opcionales
pip install wittgenlab[full]

# Instalación para desarrollo
git clone https://github.com/yourusername/wittgenlab.git
cd wittgenlab
poetry install
```

### Uso Básico

```python
from wittgenlab import EvalHub

# Inicializar el hub de evaluación
evaluator = EvalHub()

# Evaluar salidas de modelos con múltiples métricas
results = evaluator.evaluate(
    predictions=model_outputs,
    references=ground_truth,
    metrics=['bleu', 'bertscore', 'rouge'],
    task='summarization'
)

print(f"BLEU Score: {results.get_score('bleu'):.4f}")
print(f"BERTScore: {results.get_score('bertscore'):.4f}")
```

### BERTScore - Evaluación Semántica

```python
# BERTScore con configuraciones completas
results = evaluator.evaluate(
    predictions=predictions,
    references=references,
    metrics=['bertscore'],
    task='semantic_evaluation',
    return_full_scores=True,  # Obtener precisión, recall, F1
    model_type="bert-base-multilingual-cased"
)

bertscore_result = results.get_score('bertscore')
print(f"Precisión: {bertscore_result['precision']:.4f}")
print(f"Recall: {bertscore_result['recall']:.4f}")
print(f"F1: {bertscore_result['f1']:.4f}")
```

### LLM-as-a-Judge

```python
# Juez con múltiples modelos y criterios
judge_results = evaluator.judge(
    predictions=outputs,
    criteria=['accuracy', 'helpfulness', 'safety'],
    judge_models=['gpt-4', 'claude-3', 'gemini-pro'],
    consensus_method='majority_vote'
)

# Acceder a resultados por criterio
for criterion, stats in judge_results['summary'].items():
    print(f"{criterion}: {stats['mean_score']:.2f}/5")
```

### Evaluación de Benchmarks

```python
# Ejecutar benchmarks estandarizados
benchmark_results = evaluator.benchmark(
    model=my_model,
    benchmarks=['mmlu', 'hellaswag', 'truthfulqa'],
    few_shot=5
)

print(f"MMLU Score: {benchmark_results.get_score('mmlu'):.4f}")
```

### Comparación de Modelos

```python
# Comparar múltiples modelos
models = {
    'gpt-4': gpt4_model,
    'claude-3': claude_model,
    'llama-2': llama_model
}

comparison = evaluator.compare_models(
    models=models,
    benchmarks=['mmlu', 'hellaswag'],
    few_shot=3
)

# Obtener rankings
ranking = comparison.get_overall_ranking()
print(f"Mejor modelo: {ranking[0][0]} con puntuación {ranking[0][1]:.4f}")
```

## 📁 Estructura del Framework

```
wittgenlab/
├── metrics/           # Métricas de evaluación
│   ├── ngram/        # BLEU, ROUGE, METEOR, CIDEr
│   ├── semantic/     # BERTScore, MoverScore, BLEURT
│   ├── perplexity/   # Métricas basadas en perplejidad
│   └── custom/       # Métricas personalizadas
├── benchmarks/       # Benchmarks estandarizados
│   ├── glue/         # GLUE, SuperGLUE
│   ├── knowledge/    # MMLU, ARC, HellaSwag
│   ├── code/         # HumanEval, MBPP
│   ├── safety/       # ToxiGen, TruthfulQA
│   └── multilingual/ # Benchmarks multilingües
├── judges/           # Implementaciones LLM-as-a-Judge
│   ├── pairwise/     # Comparaciones pareadas
│   ├── scoring/      # Puntuación absoluta
│   └── reasoning/    # Juicio con cadena de pensamiento
├── human/            # Herramientas de evaluación humana
│   ├── annotation/   # Interfaces de anotación
│   ├── crowdsource/  # Integraciones de plataformas
│   └── agreement/    # Acuerdo entre anotadores
└── analysis/         # Análisis y visualización
    ├── correlation/  # Análisis de correlación
    ├── bias/         # Detección de sesgos
    └── reports/      # Generación de reportes
```

## 📊 Métricas Soportadas

### Métricas N-gram
- **BLEU**: Bilingual Evaluation Understudy
- **ROUGE**: Recall-Oriented Understudy for Gisting Evaluation
- **METEOR**: Metric for Evaluation of Translation with Explicit ORdering
- **CIDEr**: Consensus-based Image Description Evaluation

### Métricas Semánticas
- **BERTScore**: Similitud semántica consciente del contexto
- **MoverScore**: Earth Mover's Distance para evaluación
- **BLEURT**: Métrica de evaluación aprendida

### LLM-as-a-Judge
- **Criterios Múltiples**: accuracy, helpfulness, safety, quality, relevance, fluency
- **Modelos Múltiples**: GPT-4, Claude-3, Gemini-Pro, y más
- **Métodos de Consenso**: majority_vote, average, weighted_average
- **Criterios Personalizados**: Define tus propios criterios de evaluación

### Métricas Personalizadas
- **Diversidad**: Diversidad léxica y semántica
- **Coherencia**: Evaluación de coherencia textual
- **Fluidez**: Evaluación de fluidez del lenguaje

## 🏆 Benchmarks Soportados

### Conocimiento y Razonamiento
- **MMLU**: Massive Multitask Language Understanding
- **ARC**: AI2 Reasoning Challenge
- **HellaSwag**: Razonamiento de sentido común
- **WinoGrande**: Winograd Schema Challenge

### Generación de Código
- **HumanEval**: Generación de código Python
- **MBPP**: Mostly Basic Python Problems

### Seguridad y Alineación
- **ToxiGen**: Detección de toxicidad
- **TruthfulQA**: Evaluación de veracidad
- **Detección de Sesgos**: Varias suites de evaluación de sesgos

## ⚙️ Configuración

Crear configuraciones personalizadas para tus necesidades de evaluación:

```python
from wittgenlab import EvalConfig

config = EvalConfig(
    batch_size=32,
    use_cache=True,
    log_level="INFO",
    output_dir="./eval_results"
)

# Configurar métricas específicas
config.set_metric_config('bleu', {
    'max_order': 4,
    'smooth': True
})

# Configurar BERTScore
config.set_metric_config('bertscore', {
    'model_type': 'bert-base-multilingual-cased',
    'return_full_scores': True
})

evaluator = EvalHub(config=config)
```

## 🤖 LLM-as-a-Judge Avanzado

### Uso Básico

```python
from wittgenlab.judges import create_judge

# Crear un juez individual
judge = create_judge(model_name="gpt-4o-mini")

result = judge.evaluate(
    prediction="Tu texto a evaluar",
    criterion="accuracy",
    context="Contexto opcional"
)

print(f"Puntuación: {result.score}/5")
print(f"Justificación: {result.justification}")
```

### Múltiples Modelos con Consenso

```python
# Configurar múltiples modelos
judge_results = evaluator.judge(
    predictions=["Texto 1", "Texto 2"],
    criteria=['accuracy', 'helpfulness', 'safety'],
    judge_models=['gpt-4', 'claude-3', 'gemini-pro'],
    consensus_method='majority_vote'
)

# Análisis de resultados
for criterion, stats in judge_results['summary'].items():
    print(f"{criterion}: {stats['mean_score']:.2f}/5")
```

### Criterios Personalizados

```python
# Definir criterios personalizados
custom_result = judge.evaluate(
    prediction=code_snippet,
    criterion="code_quality_and_efficiency",
    context="Función para calcular Fibonacci"
)
```

## 📈 Análisis y Reportes

Generar reportes integrales de evaluación:

```python
# Generar reporte HTML
report_path = evaluator.generate_report(
    results=benchmark_results,
    format="html",
    output_path="evaluation_report.html"
)

# Analizar correlaciones entre métricas
from wittgenlab.analysis import CorrelationAnalyzer

analyzer = CorrelationAnalyzer()
correlations = analyzer.analyze(results)
```

## 🔧 Extensibilidad del Framework

### Agregar Métricas Personalizadas

```python
from wittgenlab.metrics.base import ReferenceBasedMetric

class MyCustomMetric(ReferenceBasedMetric):
    def _compute_score(self, predictions, references):
        # Tu implementación de métrica
        return score

# Registrar la métrica
evaluator.metrics_registry.register_metric("my_metric", MyCustomMetric)
```

### Agregar Benchmarks Personalizados

```python
from wittgenlab.benchmarks.base import BaseBenchmark

class MyBenchmark(BaseBenchmark):
    def load_data(self):
        # Cargar tus datos de benchmark
        return data
    
    def evaluate(self, model):
        # Lógica de evaluación
        return results

# Registrar el benchmark
evaluator.benchmarks_registry.register_benchmark("my_benchmark", MyBenchmark)
```

### Implementar Jueces Personalizados

```python
from wittgenlab.judges.base import BaseJudge, JudgeResult

class CustomJudge(BaseJudge):
    def _initialize_model(self):
        # Inicializar tu modelo personalizado
        pass
    
    def evaluate(self, prediction, reference=None, criterion="quality", **kwargs):
        # Tu lógica de evaluación personalizada
        return JudgeResult(
            score=score,
            justification="Tu justificación",
            criterion=criterion,
            model_name=self.model_name
        )
```

## 🚀 Ejemplos Completos

### Ejemplo 1: Evaluación de Resúmenes

```python
from wittgenlab import EvalHub

evaluator = EvalHub()

# Datos de ejemplo
summaries = ["Resumen generado por modelo...", ...]
references = ["Resumen de referencia...", ...]

# Evaluación combinada
traditional_results = evaluator.evaluate(
    predictions=summaries,
    references=references,
    metrics=['bleu', 'rouge', 'bertscore'],
    task='summarization'
)

llm_results = evaluator.judge(
    predictions=summaries,
    criteria=['accuracy', 'coherence', 'conciseness'],
    judge_models=['gpt-4'],
    references=references
)

print("Métricas Automáticas:")
for metric, score in traditional_results.get_all_scores().items():
    print(f"  {metric}: {score:.4f}")

print("Evaluación LLM:")
for criterion, stats in llm_results['summary'].items():
    print(f"  {criterion}: {stats['mean_score']:.2f}/5")
```

### Ejemplo 2: Comparación de Chatbots

```python
# Comparar respuestas de diferentes chatbots
chatbot_responses = {
    "ChatGPT": ["Respuesta 1", "Respuesta 2"],
    "Claude": ["Respuesta 1", "Respuesta 2"],
    "Gemini": ["Respuesta 1", "Respuesta 2"]
}

for bot_name, responses in chatbot_responses.items():
    results = evaluator.judge(
        predictions=responses,
        criteria=['helpfulness', 'safety', 'accuracy'],
        judge_models=['gpt-4', 'claude-3'],
        consensus_method='average'
    )
    
    print(f"\n{bot_name} Results:")
    for criterion, stats in results['summary'].items():
        print(f"  {criterion}: {stats['mean_score']:.2f}/5")
```

## 📚 Documentación

Para documentación detallada, ejemplos y referencia de API, visita:
- [Documentación Completa](https://wittgenlab.readthedocs.io)
- [Referencia de API](https://wittgenlab.readthedocs.io/api)
- [Ejemplos](https://github.com/yourusername/wittgenlab/tree/main/examples)

## 🎮 Ejecutar la Demostración

```bash
# Ejecutar ejemplo básico
python examples/basic_usage.py

# Ejecutar demostración avanzada
python examples/advanced_usage.py

# Ejecutar demostración completa
python examples/complete_demo.py
```

## 🤝 Contribuir

¡Damos la bienvenida a las contribuciones! Consulta nuestra [Guía de Contribución](CONTRIBUTING.md) para más detalles.

1. Haz fork del repositorio
2. Crea una rama de feature
3. Realiza tus cambios
4. Agrega tests
5. Envía un pull request

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.

## 🙏 Agradecimientos

- Construido sobre excelentes librerías como HuggingFace Transformers, SacreBLEU, BERTScore y LangChain
- Inspirado por frameworks de evaluación de la comunidad de investigación
- Gracias a todos los contribuidores y usuarios del framework

## 📞 Contacto

- **Autor**: Robert Gomez
- **Email**: robertgomez.datascience@gmail.com
- **GitHub**: [@Robert-Gomez-AI](https://github.com/Robert-Gomez-AI)

---

⭐ **¡Dale estrella a este repositorio si te resulta útil!** ⭐
