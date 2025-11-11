import json
import time
import hashlib
import threading
import re
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pydantic import Field
import numpy as np

from agent_hive.enum import ContextType
from agent_hive.task import Task
from agent_hive.workflows.base_workflow import Workflow
from agent_hive.logger import get_custom_logger
from agent_hive.agents.base_agent import BaseAgent

logger = get_custom_logger(__name__)


class SemanticMatcher:
    
    DOMAIN_KEYWORDS = {
        'forecast': ['predict', 'forecast', 'future', 'trend', 'projection', 'estimate'],
        'sensor': ['sensor', 'measurement', 'reading', 'monitor', 'signal', 'data'],
        'diagnosis': ['fault', 'failure', 'error', 'issue', 'problem', 'diagnose'],
        'maintenance': ['maintenance', 'repair', 'service', 'upkeep', 'preventive'],
        'analysis': ['analyze', 'examine', 'investigate', 'study', 'evaluate'],
        'temporal': ['time', 'period', 'duration', 'interval', 'historical'],
        'quantitative': ['number', 'value', 'measurement', 'metric', 'quantity']
    }
    
    @staticmethod
    def tokenize(text: str) -> Set[str]:
        tokens = re.findall(r'\b\w+\b', text.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        return set(t for t in tokens if len(t) > 2 and t not in stopwords)
    
    @staticmethod
    def compute_similarity(text1: str, text2: str) -> float:
        tokens1 = SemanticMatcher.tokenize(text1)
        tokens2 = SemanticMatcher.tokenize(text2)
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        jaccard = intersection / union if union > 0 else 0.0
        
        domain_score = 0.0
        for domain, keywords in SemanticMatcher.DOMAIN_KEYWORDS.items():
            in_text1 = sum(1 for kw in keywords if kw in text1.lower())
            in_text2 = sum(1 for kw in keywords if kw in text2.lower())
            if in_text1 > 0 and in_text2 > 0:
                domain_score += 0.1
        
        return min(1.0, jaccard * 0.7 + domain_score * 0.3)
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 5) -> List[str]:
        tokens = SemanticMatcher.tokenize(text)
        
        scored = []
        for token in tokens:
            score = 0
            for domain, keywords in SemanticMatcher.DOMAIN_KEYWORDS.items():
                if token in keywords:
                    score += 2
                elif any(token in kw or kw in token for kw in keywords):
                    score += 1
            scored.append((score + len(token) * 0.1, token))
        
        scored.sort(reverse=True)
        return [token for _, token in scored[:top_n]]


class CircuitBreaker:
    
    class State(Enum):
        CLOSED = "closed"      
        OPEN = "open"          
        HALF_OPEN = "half_open"  
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func, *args, **kwargs):
        with self.lock:
            if self.state == self.State.OPEN:
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    logger.info("Circuit breaker entering HALF_OPEN state")
                    self.state = self.State.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN - too many failures")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        with self.lock:
            if self.state == self.State.HALF_OPEN:
                logger.info("Circuit breaker CLOSED - service recovered")
            self.state = self.State.CLOSED
            self.failure_count = 0
    
    def _on_failure(self):
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != self.State.OPEN:
                    logger.warning(f"Circuit breaker OPEN after {self.failure_count} failures")
                self.state = self.State.OPEN


class ResponseCache:
    
    def __init__(self, max_size: int = 50, ttl: float = 3600.0):
        self.cache: Dict[str, Tuple[str, float, float]] = {}  
        self.access_order = deque()
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, agent_name: str, task_input: str) -> str:
        key = f"{agent_name}:{task_input}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
    
    def get(self, agent_name: str, task_input: str) -> Optional[Tuple[str, float]]:
        cache_key = self._compute_hash(agent_name, task_input)
        
        with self.lock:
            if cache_key in self.cache:
                response, timestamp, quality = self.cache[cache_key]
                
                if time.time() - timestamp < self.ttl:
                    if cache_key in self.access_order:
                        self.access_order.remove(cache_key)
                    self.access_order.append(cache_key)
                    
                    self.hits += 1
                    logger.info(f"Cache HIT for {agent_name} (quality: {quality:.2f})")
                    return response, quality
                else:
                    del self.cache[cache_key]
            
            self.misses += 1
            return None
    
    def put(self, agent_name: str, task_input: str, response: str, quality: float):
        cache_key = self._compute_hash(agent_name, task_input)
        
        with self.lock:
            if len(self.cache) >= self.max_size:
                if self.access_order:
                    oldest = self.access_order.popleft()
                    if oldest in self.cache:
                        del self.cache[oldest]
            
            self.cache[cache_key] = (response, time.time(), quality)
            self.access_order.append(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class EnsembleAggregator:
    
    @staticmethod
    def majority_vote(responses: List[Tuple[str, float]], threshold: float = 0.6) -> Optional[Tuple[str, float]]:
        if not responses:
            return None
        
        if len(responses) == 1:
            return responses[0]
        
        similarity_matrix = []
        for i, (resp1, qual1) in enumerate(responses):
            row = []
            for j, (resp2, qual2) in enumerate(responses):
                if i == j:
                    row.append(1.0)
                else:
                    sim = SemanticMatcher.compute_similarity(resp1, resp2)
                    row.append(sim)
            similarity_matrix.append(row)
        
        scores = []
        for i, (resp, qual) in enumerate(responses):
            avg_similarity = sum(similarity_matrix[i]) / len(similarity_matrix[i])
            combined_score = 0.6 * avg_similarity + 0.4 * qual
            scores.append((combined_score, resp, qual))
        
        scores.sort(reverse=True)
        best_score, best_response, best_quality = scores[0]
        
        logger.info(f"Ensemble consensus: {best_score:.2f} (from {len(responses)} responses)")
        
        return best_response, best_quality
    
    @staticmethod
    def weighted_merge(responses: List[Tuple[str, float]]) -> Optional[Tuple[str, float]]:
        if not responses:
            return None
        
        if len(responses) == 1:
            return responses[0]
        
        responses = sorted(responses, key=lambda x: x[1], reverse=True)
        
        best_response, best_quality = responses[0]
        
        merged_parts = [best_response]
        
        for resp, qual in responses[1:]:
            if qual >= 0.5:  
                similarity = SemanticMatcher.compute_similarity(best_response, resp)
                if similarity < 0.7:  
                    sentences = [s.strip() for s in resp.split('.') if len(s.strip()) > 20]
                    for sent in sentences[:2]:  
                        if not any(SemanticMatcher.compute_similarity(sent, merged_parts[0]) > 0.6 for _ in [1]):
                            merged_parts.append(sent)
        
        merged = ". ".join(merged_parts)
        avg_quality = sum(q for _, q in responses) / len(responses)
        
        return merged, avg_quality


@dataclass
class AgentMetrics:
    name: str
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    response_qualities: List[float] = field(default_factory=list)
    task_type_scores: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    consecutive_failures: int = 0
    last_success_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5
    
    @property
    def avg_time(self) -> float:
        total = self.successes + self.failures
        return self.total_time / total if total > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        return sum(self.response_qualities) / len(self.response_qualities) if self.response_qualities else 0.5
    
    def get_task_type_score(self, task_type: str) -> float:
        scores = self.task_type_scores.get(task_type, [])
        return sum(scores) / len(scores) if scores else 0.5
    
    def is_healthy(self) -> bool:
        return self.consecutive_failures < 3 and self.success_rate > 0.3


class ResponseValidator:
    
    REFUSAL_PATTERNS = [
        r"i (don't|do not|cannot|can't) (have access|access|know|provide)",
        r"as an ai",
        r"i (apologize|am sorry)",
        r"unfortunately",
        r"i'm (not sure|unable|sorry)",
        r"cannot determine",
        r"not enough (information|data)",
    ]
    
    CONFIDENCE_PATTERNS = [
        r"\b(indicates?|suggests?|shows?|demonstrates?)\b",
        r"\b(likely|probable|expected|anticipated)\b",
        r"\b(based on|according to|analysis shows?)\b",
        r"\b(confident|certain|clear)\b",
    ]
    
    @staticmethod
    def validate(response: str, task_desc: str, task_type: str) -> Tuple[bool, float, Dict[str, Any]]:
        details = {
            "length_check": False,
            "hallucination_check": False,
            "relevance_check": False,
            "structure_check": False,
            "confidence_check": False,
            "task_specific_check": False
        }
        
        if not response or len(response.strip()) < 20:
            return False, 0.0, details
        
        score = 0.0
        resp_lower = response.lower()
        
        length = len(response.strip())
        if 100 <= length <= 2000:
            score += 0.20
            details["length_check"] = True
        elif 30 <= length < 100:
            score += 0.10
        elif length > 2000:
            score += 0.05  
        
        has_refusal = any(re.search(pattern, resp_lower) for pattern in ResponseValidator.REFUSAL_PATTERNS)
        if has_refusal:
            score -= 0.40  
            logger.warning("Hallucination detected: refusal pattern found")
        else:
            score += 0.25
            details["hallucination_check"] = True
        
        similarity = SemanticMatcher.compute_similarity(task_desc, response)
        if similarity >= 0.3:
            score += 0.15
            details["relevance_check"] = True
        elif similarity >= 0.15:
            score += 0.08
        
        has_structure = (
            response.count('\n') >= 1 or
            any(marker in response for marker in ['1.', '2.', '- ', '* ', ': '])
        )
        if has_structure:
            score += 0.10
            details["structure_check"] = True
        
        has_confidence = any(re.search(pattern, resp_lower) for pattern in ResponseValidator.CONFIDENCE_PATTERNS)
        if has_confidence:
            score += 0.10
            details["confidence_check"] = True
        
        task_score = ResponseValidator._validate_task_specific(response, task_desc, task_type)
        score += task_score * 0.20
        if task_score >= 0.5:
            details["task_specific_check"] = True
        
        is_valid = score >= 0.45 and not has_refusal and length >= 30
        final_score = max(0.0, min(1.0, score))
        
        return is_valid, final_score, details
    
    @staticmethod
    def _validate_task_specific(response: str, task_desc: str, task_type: str) -> float:
        resp_lower = response.lower()
        task_lower = task_desc.lower()
        score = 0.0
        
        if task_type == 'forecast':
            has_numbers = bool(re.search(r'\d+\.?\d*', response))
            has_forecast_terms = any(term in resp_lower for term in ['forecast', 'predict', 'trend', 'future', 'expect'])
            if has_numbers and has_forecast_terms:
                score = 1.0
            elif has_forecast_terms:
                score = 0.5
        
        elif task_type == 'sensor':
            sensor_terms = ['sensor', 'measurement', 'reading', 'monitor', 'signal', 'data']
            matches = sum(1 for term in sensor_terms if term in resp_lower)
            score = min(matches / 3, 1.0)
        
        elif task_type == 'diagnosis':
            diagnostic_terms = ['cause', 'because', 'due to', 'reason', 'indicates', 'suggests', 'fault']
            matches = sum(1 for term in diagnostic_terms if term in resp_lower)
            score = min(matches / 3, 1.0)
        
        else:
            if '?' in task_desc:
                answer_terms = ['is', 'are', 'answer', 'result', 'shows', 'indicates', 'therefore']
                matches = sum(1 for term in answer_terms if term in resp_lower)
                score = min(matches / 2, 1.0)
            else:
                score = 0.5  
        
        return score


class RetryStrategy:
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 30.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def get_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = delay * 0.25 * (2 * np.random.random() - 1)
        return max(0.1, delay + jitter)
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        if attempt >= self.max_retries:
            return False
        
        error_msg = str(exception).lower()
        non_retryable = ['invalid', 'unauthorized', 'forbidden', 'not found']
        if any(term in error_msg for term in non_retryable):
            return False
        
        return True


class ContextManager:
    
    def __init__(self, max_context_length: int = 1500):
        self.max_context_length = max_context_length
        self.semantic_matcher = SemanticMatcher()
    
    def select_relevant_context(self, context_parts: List[str], current_task: str) -> str:
        if not context_parts:
            return ""
        
        total_length = sum(len(part) for part in context_parts)
        
        if total_length <= self.max_context_length:
            return "\n\n---\n\n".join(context_parts)
        
        scored_contexts = []
        
        for i, part in enumerate(context_parts):
            similarity = self.semantic_matcher.compute_similarity(current_task, part)
            
            recency_score = (i + 1) / len(context_parts)
            
            length_score = 1.0 if len(part) < 500 else 0.6
            
            composite = 0.5 * similarity + 0.3 * recency_score + 0.2 * length_score
            
            scored_contexts.append((composite, i, part))
        
        scored_contexts.sort(reverse=True)
        
        selected = []
        current_length = 0
        
        for score, idx, part in scored_contexts:
            if current_length + len(part) <= self.max_context_length:
                selected.append((idx, part))
                current_length += len(part)
            elif current_length < self.max_context_length * 0.8:
                remaining = self.max_context_length - current_length
                truncated = part[:remaining-10] + "...[truncated]"
                selected.append((idx, truncated))
                break
        
        selected.sort(key=lambda x: x[0])
        
        result = "\n\n---\n\n".join(part for _, part in selected)
        
        logger.info(f"Context pruning: {len(context_parts)}→{len(selected)} parts, "
                   f"{total_length}→{len(result)} chars")
        
        return result


class DAGAnalyzer:
    
    @staticmethod
    def get_execution_order(tasks: List[Task]) -> List[List[int]]:
        n = len(tasks)
        dependencies = {i: set() for i in range(n)}
        dependents = {i: set() for i in range(n)}
        
        for i, task in enumerate(tasks):
            if hasattr(task, 'context') and isinstance(task.context, list):
                for dep_task in task.context:
                    if dep_task in tasks:
                        dep_idx = tasks.index(dep_task)
                        if dep_idx < i:
                            dependencies[i].add(dep_idx)
                            dependents[dep_idx].add(i)
        
        completed = set()
        groups = []
        
        while len(completed) < n:
            ready = [
                i for i in range(n)
                if i not in completed and dependencies[i].issubset(completed)
            ]
            
            if not ready:
                logger.error("Circular dependency detected in task graph")
                remaining = [i for i in range(n) if i not in completed]
                groups.extend([[i] for i in remaining])
                break
            
            groups.append(ready)
            completed.update(ready)
        
        logger.info(f"DAG analysis: {n} tasks → {len(groups)} parallel groups")
        return groups


class TaskRevisionHelperAgent(BaseAgent):
    
    name = "TaskRevisionHelperAgent"
    description = "Revises and optimizes task inputs with domain-specific enhancement."
    memory = []
    tools = []

    def __init__(self, llm: str = None, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
        self.semantic_matcher = SemanticMatcher()

    def execute_task(self, task_input: str) -> str:
        task_lower = task_input.lower()
        
        keywords = self.semantic_matcher.extract_keywords(task_input)
        
        domains = []
        domain_mapping = {
            'forecast': ['forecast', 'predict', 'future', 'trend'],
            'sensor': ['sensor', 'monitor', 'measurement', 'signal'],
            'diagnosis': ['fault', 'failure', 'diagnose', 'issue', 'error'],
            'maintenance': ['maintenance', 'repair', 'service', 'upkeep'],
            'analysis': ['analyze', 'examine', 'investigate', 'evaluate'],
            'root_cause': ['root cause', 'why', 'reason', 'cause'],
        }
        
        for domain, indicators in domain_mapping.items():
            if any(ind in task_lower for ind in indicators):
                domains.append(domain)
        
        primary_domain = domains[0] if domains else 'general'
        
        enhanced = f"**PRIMARY OBJECTIVE:** {task_input.strip()}\n\n"
        
        if keywords:
            enhanced += f"**KEY FOCUS AREAS:** {', '.join(keywords)}\n\n"
        
        if primary_domain == 'forecast':
            enhanced += (
                "**FORECASTING FRAMEWORK:**\n"
                "1. Analyze historical patterns and identify trends\n"
                "2. Consider seasonality, cycles, and anomalies\n"
                "3. Provide numerical predictions with time horizons\n"
                "4. State confidence level (high/medium/low) with reasoning\n"
                "5. Identify key assumptions and limitations\n\n"
            )
        
        elif primary_domain == 'sensor':
            enhanced += (
                "**SENSOR ANALYSIS PROTOCOL:**\n"
                "1. Identify specific sensor types and their measurement targets\n"
                "2. Evaluate sensor placement and coverage\n"
                "3. Assess measurement accuracy and reliability factors\n"
                "4. Explain correlation with system behavior/failures\n"
                "5. Consider sensor fusion if multiple sensors involved\n\n"
            )
        
        elif primary_domain == 'diagnosis':
            enhanced += (
                "**DIAGNOSTIC REASONING FRAMEWORK:**\n"
                "1. Identify observable symptoms and failure modes\n"
                "2. Apply root cause analysis (5 Whys or Fishbone)\n"
                "3. Consider equipment specifications and operating conditions\n"
                "4. Rank potential causes by likelihood and impact\n"
                "5. Provide evidence-based conclusions\n\n"
            )
        
        elif primary_domain == 'root_cause':
            enhanced += (
                "**ROOT CAUSE ANALYSIS:**\n"
                "1. Trace symptom → immediate cause → underlying cause\n"
                "2. Use logical reasoning with cause-effect chains\n"
                "3. Consider multiple contributing factors\n"
                "4. Distinguish correlation from causation\n"
                "5. Provide actionable insights\n\n"
            )
        
        elif primary_domain == 'maintenance':
            enhanced += (
                "**MAINTENANCE ANALYSIS:**\n"
                "1. Assess current condition and degradation indicators\n"
                "2. Predict remaining useful life if applicable\n"
                "3. Recommend maintenance actions with priorities\n"
                "4. Consider cost-benefit and risk factors\n"
                "5. Suggest preventive measures\n\n"
            )
        
        enhanced += (
            "**EXECUTION REQUIREMENTS (MANDATORY):**\n"
            "✓ Provide DIRECT, ACTIONABLE answers - NO hedging or disclaimers\n"
            "✓ Base ALL statements on context/data provided or logical inference\n"
            "✓ If making assumptions, state them explicitly and proceed\n"
            "✓ Use specific examples, numbers, and data points when available\n"
            "✓ Structure response clearly: conclusion first, then reasoning\n"
            "✓ Express uncertainty quantitatively (e.g., 'confidence: 85%')\n"
            "✓ NEVER use phrases like 'I cannot access', 'I don't have', or 'as an AI'\n\n"
            "**OUTPUT STRUCTURE:**\n"
            "- Direct answer/conclusion (1-2 sentences)\n"
            "- Supporting analysis (2-4 bullet points)\n"
            "- Confidence assessment and limitations\n"
            "- Keep total response: 150-600 words\n\n"
            f"**DOMAIN TAGS:** [{primary_domain}] {', '.join(domains[1:]) if len(domains) > 1 else ''}\n"
        )
        
        return enhanced


class DynamicWorkflow(Workflow):
    
    context_type: ContextType = Field(
        default=ContextType.DISABLED, description="Type of context to use."
    )

    def __init__(
        self,
        tasks: List[Task],
        context_type: ContextType = ContextType.DISABLED,
        max_memory: int = 10,
    ):
        self.tasks = tasks
        self.context_type = context_type
        self.memory: List[str] = []
        self.max_memory = max_memory
        
        self.agent_metrics: Dict[str, AgentMetrics] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.response_cache = ResponseCache(max_size=50, ttl=1800.0)
        self.validator = ResponseValidator()
        self.context_mgr = ContextManager(max_context_length=1500)
        self.retry_strategy = RetryStrategy(max_retries=2)
        self.revision_agent = TaskRevisionHelperAgent()
        self.ensemble_aggregator = EnsembleAggregator()
        
        self.start_time = None
        self.task_execution_times: Dict[int, float] = {}
        self.lock = threading.Lock()
        
        self.execution_log: List[Dict[str, Any]] = []
        
        self._verify_tasks()

    def _verify_tasks(self):
        if not isinstance(self.tasks, list):
            raise ValueError("tasks must be a list of Task objects")

        for i, task in enumerate(self.tasks):
            if not task.agents:
                raise ValueError("Task must have at least one agent")

            if len(task.agents) > 1:
                logger.info(f"Task {i+1}: Multi-agent ({len(task.agents)} agents) - ensemble mode enabled")

            if self.context_type == ContextType.SELECTED and isinstance(task.context, list):
                for context_task in task.context:
                    if context_task not in self.tasks[:i]:
                        raise ValueError("Invalid context dependencies")

    def _get_circuit_breaker(self, agent_name: str) -> CircuitBreaker:
        if agent_name not in self.circuit_breakers:
            self.circuit_breakers[agent_name] = CircuitBreaker(
                failure_threshold=3,
                recovery_timeout=30.0
            )
        return self.circuit_breakers[agent_name]

    def _get_agent_metrics(self, agent_name: str) -> AgentMetrics:
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentMetrics(name=agent_name)
        return self.agent_metrics[agent_name]

    def _select_best_agent(self, agents: List[BaseAgent], task_type: str) -> BaseAgent:
        if len(agents) == 1:
            return agents[0]
        
        scored_agents = []
        
        for agent in agents:
            metrics = self._get_agent_metrics(agent.name)
            
            if not metrics.is_healthy():
                logger.warning(f"Agent {agent.name} unhealthy (consecutive failures: {metrics.consecutive_failures})")
                score = 0.1  
            else:
                success_score = metrics.success_rate
                
                task_score = metrics.get_task_type_score(task_type)
                
                speed_score = 1.0 / (1.0 + metrics.avg_time / 60.0)
                
                quality_score = metrics.avg_quality
                
                score = (0.4 * success_score + 0.3 * task_score + 
                        0.2 * speed_score + 0.1 * quality_score)
            
            scored_agents.append((score, agent))
        
        scored_agents.sort(reverse=True)
        best_agent = scored_agents[0][0]
        
        logger.info(f"Agent selection for {task_type}: {best_agent.name} (score: {scored_agents[0][0]:.3f})")
        
        return best_agent

    def _execute_agent_with_protection(
        self, 
        agent: BaseAgent, 
        task_input: str,
        task_desc: str,
        task_type: str,
        timeout: float
    ) -> Optional[Tuple[str, float]]:
        cached = self.response_cache.get(agent.name, task_input)
        if cached:
            return cached
        
        circuit_breaker = self._get_circuit_breaker(agent.name)
        metrics = self._get_agent_metrics(agent.name)
        
        start_time = time.time()
        
        try:
            response = circuit_breaker.call(agent.execute_task, task_input)
            exec_time = time.time() - start_time
            
            if not response:
                raise Exception("Empty response")
            
            response = response.replace("Final Answer:", "").strip()
            
            is_valid, quality, details = self.validator.validate(response, task_desc, task_type)
            
            if not is_valid:
                logger.warning(f"Agent {agent.name} response validation failed: {details}")
                raise Exception(f"Validation failed: quality={quality:.2f}")
            
            with self.lock:
                metrics.successes += 1
                metrics.total_time += exec_time
                metrics.response_qualities.append(quality)
                metrics.task_type_scores[task_type].append(quality)
                metrics.consecutive_failures = 0
                metrics.last_success_time = time.time()
            
            self.response_cache.put(agent.name, task_input, response, quality)
            
            logger.info(f"Agent {agent.name} success: quality={quality:.2f}, time={exec_time:.1f}s")
            
            return response, quality
            
        except Exception as e:
            exec_time = time.time() - start_time
            
            with self.lock:
                metrics.failures += 1
                metrics.total_time += exec_time
                metrics.consecutive_failures += 1
            
            logger.error(f"Agent {agent.name} failed: {str(e)}")
            return None

    def _execute_with_retry(
        self,
        agent: BaseAgent,
        task_input: str,
        task_desc: str,
        task_type: str,
        base_timeout: float
    ) -> Optional[Tuple[str, float]]:
        for attempt in range(self.retry_strategy.max_retries + 1):
            timeout = base_timeout * (0.8 ** attempt)
            
            logger.info(f"Agent {agent.name} attempt {attempt + 1}/{self.retry_strategy.max_retries + 1}")
            
            result = self._execute_agent_with_protection(
                agent, task_input, task_desc, task_type, timeout
            )
            
            if result:
                return result
            
            if attempt < self.retry_strategy.max_retries:
                delay = self.retry_strategy.get_delay(attempt)
                logger.info(f"Retry after {delay:.2f}s delay")
                time.sleep(delay)
        
        return None

    def _execute_with_ensemble(
        self,
        agents: List[BaseAgent],
        task_input: str,
        task_desc: str,
        task_type: str,
        timeout: float
    ) -> Optional[Tuple[str, float]]:
        if len(agents) == 1:
            return self._execute_with_retry(agents[0], task_input, task_desc, task_type, timeout)
        
        logger.info(f"Ensemble mode: {len(agents)} agents")
        
        responses = []
        
        with ThreadPoolExecutor(max_workers=min(len(agents), 3)) as executor:
            futures = {
                executor.submit(
                    self._execute_with_retry,
                    agent, task_input, task_desc, task_type, timeout
                ): agent
                for agent in agents
            }
            
            for future in as_completed(futures, timeout=timeout * 1.5):
                agent = futures[future]
                try:
                    result = future.result(timeout=5.0)
                    if result:
                        responses.append(result)
                        logger.info(f"Ensemble: {agent.name} contributed (quality: {result[1]:.2f})")
                except Exception as e:
                    logger.warning(f"Ensemble: {agent.name} failed: {e}")
        
        if not responses:
            return None
        
        if len(responses) == 1:
            return responses[0]
        
        best_response, consensus_quality = self.ensemble_aggregator.majority_vote(responses)
        
        logger.info(f"Ensemble consensus: {len(responses)} responses, quality={consensus_quality:.2f}")
        
        return best_response, consensus_quality

    def _detect_task_type(self, desc: str) -> str:
        d = desc.lower()
        
        if 'forecast' in d or 'predict' in d or 'future' in d:
            return 'forecast'
        if 'root cause' in d or 'why' in d:
            return 'root_cause'
        if 'fault' in d or 'diagnos' in d or 'failure' in d:
            return 'diagnosis'
        if 'sensor' in d or 'monitor' in d:
            return 'sensor'
        if 'maintenance' in d or 'repair' in d:
            return 'maintenance'
        if 'analyz' in d or 'examine' in d:
            return 'analysis'
        
        return 'general'

    def _execute_single_task(self, task_idx: int) -> Optional[str]:
        task = self.tasks[task_idx]
        task_no = task_idx + 1
        
        task_start = time.time()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TASK {task_no}: {task.description[:80]}...")
        logger.info(f"{'='*60}")
        
        task_type = self._detect_task_type(task.description)
        logger.info(f"Task type: {task_type}")
        
        user_input = self._build_input(task, task_idx)
        
        try:
            enhanced_input = self.revision_agent.execute_task(user_input)
            logger.info("Task revision applied")
        except Exception as e:
            logger.warning(f"Task revision failed: {e}")
            enhanced_input = user_input
        
        agents = task.agents
        timeout = 90.0
        
        use_ensemble = (
            len(agents) > 1 and
            task_type in ['diagnosis', 'root_cause', 'forecast']  
        )
        
        if use_ensemble:
            result = self._execute_with_ensemble(
                agents, enhanced_input, task.description, task_type, timeout
            )
        else:
            best_agent = self._select_best_agent(agents, task_type)
            result = self._execute_with_retry(
                best_agent, enhanced_input, task.description, task_type, timeout
            )
            
            if not result and len(agents) > 1:
                logger.warning("Primary agent failed, trying fallback")
                fallback_agents = [a for a in agents if a != best_agent]
                for fallback_agent in fallback_agents:
                    result = self._execute_with_retry(
                        fallback_agent, enhanced_input, task.description, task_type, timeout * 0.7
                    )
                    if result:
                        break
        
        if not result:
            logger.error(f"Task {task_no} failed after all attempts")
            response = f"[Task execution failed: Unable to generate valid response]"
            quality = 0.0
        else:
            response, quality = result
        
        exec_time = time.time() - task_start
        self.task_execution_times[task_idx] = exec_time
        
        self.execution_log.append({
            "task_number": task_no,
            "task_type": task_type,
            "execution_time": exec_time,
            "quality": quality,
            "agents_used": [a.name for a in agents],
            "success": quality > 0.0
        })
        
        logger.info(f"Task {task_no} completed: time={exec_time:.1f}s, quality={quality:.2f}")
        
        return response

    def run(self):
        self.memory = []
        self.start_time = time.time()
        self.context_type = ContextType.SELECTED
        
        logger.info("\n" + "="*70)
        logger.info("DYNAMIC WORKFLOW - PRODUCTION MODE")
        logger.info(f"Tasks: {len(self.tasks)}")
        logger.info(f"Context: {self.context_type.value}")
        logger.info("="*70 + "\n")
        
        execution_groups = DAGAnalyzer.get_execution_order(self.tasks)
        logger.info(f"Execution plan: {len(execution_groups)} groups")
        
        iteration_count = 0
        max_iterations = 15
        
        for group_idx, task_indices in enumerate(execution_groups):
            if iteration_count >= max_iterations:
                logger.warning(f"Max iterations ({max_iterations}) reached - stopping")
                break
            
            logger.info(f"\n{'*'*60}")
            logger.info(f"GROUP {group_idx + 1}/{len(execution_groups)}: Tasks {[i+1 for i in task_indices]}")
            logger.info(f"{'*'*60}")
            
            if len(task_indices) == 1:
                idx = task_indices[0]
                response = self._execute_single_task(idx)
                
                while len(self.memory) <= idx:
                    self.memory.append("")
                self.memory[idx] = response or ""
                
                iteration_count += 1
            
            else:
                logger.info(f"Parallel execution: {len(task_indices)} tasks")
                
                with ThreadPoolExecutor(max_workers=min(len(task_indices), 4)) as executor:
                    future_to_idx = {
                        executor.submit(self._execute_single_task, idx): idx
                        for idx in task_indices
                    }
                    
                    for future in as_completed(future_to_idx, timeout=180.0):
                        idx = future_to_idx[future]
                        
                        try:
                            response = future.result(timeout=10.0)
                            
                            while len(self.memory) <= idx:
                                self.memory.append("")
                            self.memory[idx] = response or ""
                            
                            logger.info(f"Task {idx + 1} completed in parallel")
                            
                        except FutureTimeoutError:
                            logger.error(f"Task {idx + 1} timed out in parallel execution")
                            while len(self.memory) <= idx:
                                self.memory.append("")
                            self.memory[idx] = "[Execution timeout]"
                            
                        except Exception as e:
                            logger.error(f"Task {idx + 1} failed in parallel: {e}")
                            while len(self.memory) <= idx:
                                self.memory.append("")
                            self.memory[idx] = f"[Execution error: {str(e)[:100]}]"
                        
                        iteration_count += 1
                        if iteration_count >= max_iterations:
                            logger.warning("Max iterations reached during parallel execution")
                            break
        
        total_time = time.time() - self.start_time
        successful_tasks = sum(1 for log in self.execution_log if log['success'])
        avg_quality = sum(log['quality'] for log in self.execution_log) / len(self.execution_log) if self.execution_log else 0.0
        
        logger.info("\n" + "="*70)
        logger.info("WORKFLOW COMPLETE - PERFORMANCE SUMMARY")
        logger.info("="*70)
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Tasks completed: {successful_tasks}/{len(self.tasks)}")
        logger.info(f"Average quality: {avg_quality:.2f}")
        logger.info(f"Parallel groups: {len(execution_groups)}")
        
        logger.info("\nAgent Performance:")
        for agent_name, metrics in self.agent_metrics.items():
            logger.info(f"  {agent_name}: {metrics.success_rate:.1%} success, "
                       f"{metrics.avg_time:.1f}s avg, {metrics.avg_quality:.2f} quality")
        
        cache_stats = self.response_cache.get_stats()
        logger.info(f"\nCache: {cache_stats['hit_rate']:.1%} hit rate ({cache_stats['hits']}/{cache_stats['hits'] + cache_stats['misses']})")
        
        logger.info("="*70 + "\n")
        
        history = self.generate_history()
        print(json.dumps(history, indent=4))
        
        return history

    def _build_input(self, task: Task, idx: int) -> str:
        if self.context_type == ContextType.DISABLED:
            return task.description

        elif self.context_type == ContextType.ALL:
            context_parts = [m for m in self.memory[-self.max_memory:] if m]
            if not context_parts:
                return task.description
            
            context = self.context_mgr.select_relevant_context(context_parts, task.description)
            return f"{task.description}\n\n**Context from Previous Tasks:**\n{context}"

        elif self.context_type == ContextType.PREVIOUS:
            if not self.memory or not self.memory[-1]:
                return task.description
            return f"{task.description}\n\n**Previous Task Output:**\n{self.memory[-1]}"

        elif self.context_type == ContextType.SELECTED:
            context_tasks = task.context or []
            if not context_tasks:
                return task.description
            
            context_parts = []
            for ctx_task in context_tasks:
                try:
                    ctx_idx = self.tasks.index(ctx_task)
                    if ctx_idx < len(self.memory) and self.memory[ctx_idx]:
                        context_parts.append(
                            f"**Task {ctx_idx + 1} Output:**\n{self.memory[ctx_idx]}"
                        )
                except (ValueError, IndexError) as e:
                    logger.warning(f"Context task not found: {e}")
                    continue
            
            if not context_parts:
                return task.description
            
            context = self.context_mgr.select_relevant_context(context_parts, task.description)
            return f"{task.description}\n\n**Relevant Context:**\n{context}"

        else:
            raise ValueError(f"Invalid context_type: {self.context_type}")

    def generate_history(self):
        history = []
        
        for i, task in enumerate(self.tasks):
            exec_log = next((log for log in self.execution_log if log['task_number'] == i + 1), None)
            
            task_info = {
                "task_number": i + 1,
                "task_description": task.description,
                "agent_names": [agent.name for agent in task.agents],
                "response": self.memory[i] if i < len(self.memory) else None,
            }
            
            if exec_log:
                task_info.update({
                    "execution_time": exec_log['execution_time'],
                    "quality_score": exec_log['quality'],
                    "task_type": exec_log['task_type']
                })
            
            history.append(task_info)
        
        return history
