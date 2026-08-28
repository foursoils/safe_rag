import logging
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from safe_rag.attacks.agea.agea_prompts import (
    GRAPH_FILTER_AGENT_SYSTEM_PROMPT,
    GRAPH_FILTER_PROMPT_TEMPLATE,
)
from safe_rag.attacks.agea.graph_extractor_memory import GraphExtractorMemory
from safe_rag.attacks.agea.graph_query_memory import QueryMemory
from safe_rag.attacks.agea.llm import get_openai_client, graph_filter_deployment
from safe_rag.attacks.agea.utils import (
    normalize_node_label,
    parse_keep_discard_decisions as util_parse_keep_discard_decisions,
    parse_llm_response_for_graph_items as util_parse_llm_response_for_graph_items,
)


logger = logging.getLogger(__name__)


def parse_llm_response_for_graph_items(llm_response: str) -> Tuple[List[dict], List[dict]]:
    return util_parse_llm_response_for_graph_items(llm_response, normalize_and_dedupe=True)


class GraphFilterAgentContext:
    def __init__(self, query_memory: QueryMemory, filtered_graph_memory: GraphExtractorMemory):
        self.query_memory = query_memory
        self.filtered_graph_memory = filtered_graph_memory
        self.graph_entity_labels: set[str] = set()
        self.graph_edge_tuples: set[Tuple[str, str]] = set()

    @staticmethod
    def _normalize_edge_tuple(edge: Any) -> Tuple[str, str]:
        source = ""
        target = ""
        if isinstance(edge, dict):
            source = normalize_node_label(str(edge.get("source") or edge.get("src") or ""))
            target = normalize_node_label(str(edge.get("target") or edge.get("dst") or ""))
        elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
            source = normalize_node_label(str(edge[0]))
            target = normalize_node_label(str(edge[1]))
        if not source or not target:
            return "", ""
        return source, target

    def update_graph_state(self) -> None:
        graph = getattr(self.filtered_graph_memory, "G", None)
        if graph is None:
            self.graph_entity_labels = set()
            self.graph_edge_tuples = set()
            return
        self.graph_entity_labels = {normalize_node_label(label) for label in graph.nodes()}
        self.graph_edge_tuples = {
            (normalize_node_label(src), normalize_node_label(dst)) for src, dst in graph.edges()
        }

    def get_graph_context(
        self,
        candidate_nodes: List[Dict[str, Any]],
        candidate_edges: List[Dict[str, Any]],
    ) -> str:
        self.update_graph_state()
        if not self.graph_entity_labels:
            return "GRAPH CONTEXT: Empty graph (this is the first turn).\n"

        candidate_labels = {
            normalize_node_label(node.get("label", node.get("id", ""))) for node in candidate_nodes
        }
        candidate_edge_tuples: set[Tuple[str, str]] = set()
        for edge in candidate_edges:
            edge_tuple = self._normalize_edge_tuple(edge)
            if all(edge_tuple):
                candidate_edge_tuples.add(edge_tuple)

        already_in_graph_nodes = candidate_labels & self.graph_entity_labels
        already_in_graph_edges = candidate_edge_tuples & self.graph_edge_tuples
        context_parts = [
            f"GRAPH CONTEXT: Graph currently has {len(self.graph_entity_labels)} entities and {len(self.graph_edge_tuples)} edges."
        ]
        if already_in_graph_nodes:
            context_parts.append(
                f"\nEntities already in graph ({len(already_in_graph_nodes)}): {', '.join(list(already_in_graph_nodes)[:10])}"
            )
            if len(already_in_graph_nodes) > 10:
                context_parts.append(f" (and {len(already_in_graph_nodes) - 10} more)")
            context_parts.append("\n→ These entities are likely valid if they reappear. Be more lenient.")
        if already_in_graph_edges:
            context_parts.append(
                f"\nEdges already in graph ({len(already_in_graph_edges)}): These relationships have been graph-filtered before."
            )
            context_parts.append("\n→ These edges are likely valid if they reappear. Be more lenient.")
        if not already_in_graph_nodes and not already_in_graph_edges:
            context_parts.append("\n→ All candidates are new. Filter carefully.")
        return "\n".join(context_parts) + "\n"

    def filter_duplicates_before_graph_filter(
        self,
        candidate_nodes: List[Dict[str, Any]],
        candidate_edges: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        self.update_graph_state()
        new_nodes: List[Dict[str, Any]] = []
        duplicate_nodes: List[Dict[str, Any]] = []
        new_edges: List[Dict[str, Any]] = []
        duplicate_edges: List[Dict[str, Any]] = []
        for node in candidate_nodes:
            label = normalize_node_label(node.get("label", node.get("id", "")))
            if label in self.graph_entity_labels:
                duplicate_nodes.append(node)
            else:
                new_nodes.append(node)
        for edge in candidate_edges:
            edge_tuple = self._normalize_edge_tuple(edge)
            if not all(edge_tuple):
                continue
            if edge_tuple in self.graph_edge_tuples:
                duplicate_edges.append(edge)
            else:
                new_edges.append(edge)
        return new_nodes, duplicate_nodes, new_edges, duplicate_edges

    def _analyze_graph_patterns(self) -> Dict[str, Any]:
        if not hasattr(self.filtered_graph_memory, "G") or len(self.filtered_graph_memory.G.nodes) == 0:
            return {}
        degrees = [self.filtered_graph_memory.G.degree(node) for node in self.filtered_graph_memory.G.nodes()]
        if not degrees:
            return {}
        avg_degree = sum(degrees) / len(degrees)
        max_degree = max(degrees)
        high_degree_threshold = max(avg_degree * 3, 10)
        high_degree_nodes = [
            node
            for node in self.filtered_graph_memory.G.nodes()
            if self.filtered_graph_memory.G.degree(node) > high_degree_threshold
        ]
        fanout_counts: Dict[str, int] = defaultdict(int)
        for source, target in self.filtered_graph_memory.G.edges():
            fanout_counts[source] += 1
        high_fanout_nodes = [node for node, count in fanout_counts.items() if count > high_degree_threshold]
        return {
            "avg_degree": avg_degree,
            "max_degree": max_degree,
            "high_degree_nodes": high_degree_nodes[:10],
            "high_fanout_nodes": high_fanout_nodes[:10],
            "high_degree_threshold": high_degree_threshold,
        }

    def _detect_suspicious_patterns(self, _turn_nodes: List[Dict[str, Any]], turn_edges: List[Any]) -> List[str]:
        warnings: List[str] = []
        node_connection_counts: Counter = Counter()
        for edge in turn_edges:
            source, target = self._normalize_edge_tuple(edge)
            if not all((source, target)):
                continue
            node_connection_counts[source] += 1
            node_connection_counts[target] += 1
        suspicious_nodes = [node for node, count in node_connection_counts.items() if count > 10]
        if suspicious_nodes:
            warnings.append(
                f"NOTE: These nodes have many new connections this turn: {', '.join(suspicious_nodes[:5])}. "
                "These might be valid hub entities - ensure they're supported by the text, but be lenient."
            )
        patterns = self._analyze_graph_patterns()
        if patterns.get("high_degree_nodes"):
            warnings.append(
                f"NOTE: These nodes already have high connectivity: {', '.join(patterns['high_degree_nodes'][:3])}. "
                "High connectivity is normal for important entities - keep connections if they're supported by the text."
            )
        return warnings

    def get_filter_guidance(
        self,
        _turn_idx: int,
        turn_nodes: Optional[List[Dict[str, Any]]] = None,
        turn_edges: Optional[List[Any]] = None,
    ) -> str:
        guidance_parts: List[str] = []
        if turn_nodes and turn_edges:
            warnings = self._detect_suspicious_patterns(turn_nodes, turn_edges)
            if warnings:
                guidance_parts.append("PATTERN WARNINGS:")
                guidance_parts.extend(warnings)
        patterns = self._analyze_graph_patterns()
        if patterns:
            if guidance_parts:
                guidance_parts.append("")
            guidance_parts.append("GRAPH STATISTICS:")
            guidance_parts.append(f"- Average node degree: {patterns.get('avg_degree', 0):.1f}")
            guidance_parts.append(f"- Max degree: {patterns.get('max_degree', 0)}")
            if patterns.get("high_degree_nodes"):
                guidance_parts.append(
                    f"- High-connectivity nodes: {len(patterns['high_degree_nodes'])} (these are likely important, not noise)"
                )
        return "\n".join(guidance_parts) if guidance_parts else ""


def format_candidates_for_graph_filter(nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]]) -> str:
    lines = ["ENTITIES:"]
    for node in nodes:
        label = node.get("label", node.get("id", ""))
        desc = node.get("description", "")[:120]
        lines.append(f"- {label} ({desc})")
    lines.append("\nRELATIONSHIPS:")
    for edge in edges:
        src = edge.get("source", edge.get("src", ""))
        dst = edge.get("target", edge.get("dst", ""))
        desc = edge.get("description", edge.get("rel", ""))[:120]
        lines.append(f"- {src} -> {dst} ({desc})")
    return "\n".join(lines)


def filter_extraction_with_graph_filter_agent(
    candidate_nodes: List[Dict[str, Any]],
    candidate_edges: List[Dict[str, Any]],
    text_content: str,
    filter_guidance: str,
    graph_context: str,
    graph_filter_model: str = "gpt-4o-mini",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    try:
        client = get_openai_client()
        deployment = graph_filter_deployment(graph_filter_model)
        prompt = GRAPH_FILTER_PROMPT_TEMPLATE.format(
            extraction_guidance=filter_guidance,
            graph_context=graph_context,
            candidate_items=format_candidates_for_graph_filter(candidate_nodes, candidate_edges),
            text_content=text_content[:4000],
        )
        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": GRAPH_FILTER_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=4096,
            temperature=0.1,
            top_p=1.0,
        )
        decision_text = response.choices[0].message.content or ""
        kept_nodes, kept_edges, found_decisions = util_parse_keep_discard_decisions(
            candidate_nodes, candidate_edges, decision_text
        )
        if not found_decisions and (candidate_nodes or candidate_edges):
            kept_nodes = candidate_nodes
            kept_edges = candidate_edges
        return kept_nodes, kept_edges, decision_text
    except Exception as exc:
        logger.warning("Graph filter agent failed: %s", exc)
        return candidate_nodes, candidate_edges, ""


def parse_and_filter_llm_response(
    llm_response: str,
    graph_filter_ctx: GraphFilterAgentContext,
    turn_idx: int,
    graph_filter_output_dir: str,
    graph_filter_model: str,
    original_nodes: Optional[set] = None,
    original_edges: Optional[set] = None,
    enable_graph_filter: bool = True,
) -> Tuple[List[dict], List[dict], List[dict], List[dict], Dict[str, Any]]:
    stats: Dict[str, Any] = {
        "raw_nodes": 0,
        "raw_edges": 0,
        "duplicate_nodes": 0,
        "duplicate_edges": 0,
        "new_nodes": 0,
        "new_edges": 0,
        "filtered_new_nodes": 0,
        "filtered_new_edges": 0,
        "filtered_nodes": 0,
        "filtered_edges": 0,
        "graph_filter_time": 0.0,
    }
    raw_nodes, raw_edges = parse_llm_response_for_graph_items(llm_response)
    stats["raw_nodes"] = len(raw_nodes)
    stats["raw_edges"] = len(raw_edges)
    if not raw_nodes and not raw_edges:
        return [], [], [], [], stats

    new_nodes, duplicate_nodes, new_edges, duplicate_edges = graph_filter_ctx.filter_duplicates_before_graph_filter(
        raw_nodes, raw_edges
    )
    stats["duplicate_nodes"] = len(duplicate_nodes)
    stats["duplicate_edges"] = len(duplicate_edges)
    stats["new_nodes"] = len(new_nodes)
    stats["new_edges"] = len(new_edges)

    filter_guidance = graph_filter_ctx.get_filter_guidance(turn_idx, raw_nodes, raw_edges)
    graph_context = graph_filter_ctx.get_graph_context(raw_nodes, raw_edges)

    if not enable_graph_filter:
        kept_new_nodes = new_nodes
        kept_new_edges = new_edges
        graph_filter_response = "Graph filter disabled - all new candidates kept."
    elif new_nodes or new_edges:
        start = time.time()
        kept_new_nodes, kept_new_edges, graph_filter_response = filter_extraction_with_graph_filter_agent(
            new_nodes,
            new_edges,
            llm_response,
            filter_guidance,
            graph_context,
            graph_filter_model=graph_filter_model,
        )
        stats["graph_filter_time"] = time.time() - start
    else:
        kept_new_nodes, kept_new_edges = [], []
        graph_filter_response = "All candidates were duplicates and auto-kept."

    filtered_nodes = kept_new_nodes + duplicate_nodes
    filtered_edges = kept_new_edges + duplicate_edges
    stats["filtered_new_nodes"] = len(kept_new_nodes)
    stats["filtered_new_edges"] = len(kept_new_edges)
    stats["filtered_nodes"] = len(filtered_nodes)
    stats["filtered_edges"] = len(filtered_edges)

    os.makedirs(graph_filter_output_dir, exist_ok=True)
    graph_filter_path = os.path.join(graph_filter_output_dir, f"graph_filter_response_query_{turn_idx}.txt")
    with open(graph_filter_path, "w", encoding="utf-8") as handle:
        handle.write(f"Turn {turn_idx} - Graph Filter Agent Output\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Graph context:\n{graph_context}\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Filter guidance:\n{filter_guidance}\n")
        handle.write("=" * 80 + "\n")
        handle.write(
            f"Raw candidates: {stats['raw_nodes']} nodes, {stats['raw_edges']} edges\n"
            f"Duplicates auto-kept: {stats['duplicate_nodes']} nodes, {stats['duplicate_edges']} edges\n"
            f"New candidates sent to filter: {stats['new_nodes']} nodes, {stats['new_edges']} edges\n"
            f"Filtered output: {stats['filtered_nodes']} nodes, {stats['filtered_edges']} edges\n"
        )
        handle.write("=" * 80 + "\n")
        handle.write(graph_filter_response)

    return raw_nodes, raw_edges, filtered_nodes, filtered_edges, stats
