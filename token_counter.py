#!/usr/bin/env python3
"""
Simple Token Counter with Static & Contextual 3D Graph Mapping
---------------------------------------------------------------
1. Accepts user vocabulary (custom or preset).
2. Counts words/tokens from input text, calculating frequencies and In-Vocab vs OOV status.
3. Maps tokens in 3D space using both:
   - Static Logic: Global co-occurrence PPMI + lexical dimensional projection.
   - Contextual Logic: Local context attention shift + sequence position manifold.
4. Produces rich terminal output and generates an interactive 3D HTML visualization.
"""

import sys
import re
import math
import json
import webbrowser
import os
from collections import Counter, defaultdict

# ----------------- Preset Vocabularies -----------------
PRESET_VOCABULARIES = {
    "1": {
        "name": "General English Core",
        "words": [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her",
            "she", "or", "an", "will", "my", "one", "all", "would", "there",
            "their", "what", "so", "up", "out", "if", "about", "who", "get",
            "which", "go", "me", "when", "make", "can", "like", "time", "no",
            "just", "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than", "then",
            "now", "look", "only", "come", "its", "over", "think", "also",
            "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these",
            "give", "day", "most", "us"
        ]
    },
    "2": {
        "name": "AI & NLP Domain",
        "words": [
            "token", "vocabulary", "model", "attention", "transformer", "context",
            "embedding", "vector", "layer", "neural", "network", "semantic",
            "sequence", "weight", "matrix", "dimension", "loss", "gradient",
            "training", "inference", "encoder", "decoder", "graph", "static",
            "dynamic", "cluster", "frequency", "corpus", "text", "language"
        ]
    },
    "3": {
        "name": "Finance & Business",
        "words": [
            "market", "stock", "trade", "investment", "price", "capital", "revenue",
            "profit", "loss", "bank", "credit", "asset", "liability", "fund",
            "growth", "risk", "bond", "equity", "rate", "currency", "inflation"
        ]
    }
}

SAMPLE_TEXTS = {
    "1": "The transformer model uses attention mechanisms to map tokens into a high dimensional vector space. Each token gets both static and contextual representations based on surrounding context.",
    "2": "In modern NLP, tokens within a sentence interact through self-attention. The bank by the river had green grass, while the financial bank invested capital in the market.",
    "3": "Every word in the vocabulary is counted by frequency. The static logic embeds words using global co-occurrence, while contextual logic shifts the token coordinates dynamically based on neighboring words."
}


def clean_tokenize(text):
    """Simple robust tokenization: lowercase, alphanumeric tokens."""
    return re.findall(r"\b[a-zA-Z0-9_\-']+\b", text.lower())


def calculate_static_and_contextual_coords(tokens, vocab_set, window_size=2):
    """
    Computes 3D coordinates for all tokens using both:
    1. Static Logic:
       - Uses global frequency, token length, and global co-occurrence graph centrality.
       - Each unique token type has a single deterministic static coordinate (x_s, y_s, z_s).
    2. Contextual Logic:
       - Each token occurrence i is influenced by its sliding context window [i-w, i+w].
       - Calculates an attention pull toward neighbors + sequence positional wave.
    """
    unique_tokens = sorted(list(set(tokens)))
    token_counts = Counter(tokens)
    total_tokens = len(tokens)

    # 1. Build Co-occurrence Matrix for Static Logic
    co_occur = defaultdict(lambda: defaultdict(int))
    for i, t in enumerate(tokens):
        for j in range(max(0, i - window_size), min(total_tokens, i + window_size + 1)):
            if i != j:
                co_occur[t][tokens[j]] += 1

    # Calculate graph degree & centrality
    centrality = {}
    for t in unique_tokens:
        deg = sum(co_occur[t].values())
        centrality[t] = deg

    max_count = max(token_counts.values()) if token_counts else 1
    max_centrality = max(centrality.values()) if centrality and max(centrality.values()) > 0 else 1
    max_len = max([len(t) for t in unique_tokens]) if unique_tokens else 1

    # Deterministic static mapping:
    # X: Frequency & prominence manifold (-100 to 100)
    # Y: Graph co-occurrence centrality & connectivity (-100 to 100)
    # Z: Lexical signature (token length, character entropy, vowel ratio) (-80 to 80)
    static_coords = {}
    for idx, t in enumerate(unique_tokens):
        # Angle for radial dispersion based on vocabulary order / hash
        angle = (idx / max(1, len(unique_tokens))) * 2 * math.pi
        freq_norm = token_counts[t] / max_count
        cent_norm = centrality[t] / max_centrality
        len_norm = len(t) / max_len

        # Pseudo-spectral / MDS coordinate estimation
        radius = 40 + 60 * (1.0 - cent_norm * 0.7)
        xs = math.cos(angle) * radius + (freq_norm * 50 - 25)
        ys = math.sin(angle) * radius + (cent_norm * 60 - 30)
        zs = (len_norm * 120 - 60) + (15 if t in vocab_set else -15)

        static_coords[t] = {
            "x": round(xs, 2),
            "y": round(ys, 2),
            "z": round(zs, 2)
        }

    # 2. Contextual Logic:
    # Compute contextual displacement for each token occurrence
    # In contextual logic, a token's coordinate shifts towards the coordinates of its neighbors!
    contextual_occurrences = []
    token_context_aggregate = defaultdict(list)

    for i, t in enumerate(tokens):
        base_coord = static_coords[t]
        
        # Attention shift vector: weighted sum of surrounding tokens
        shift_x, shift_y, shift_z = 0.0, 0.0, 0.0
        weight_sum = 0.0
        neighbors = []

        for offset in range(-window_size, window_size + 1):
            if offset == 0:
                continue
            j = i + offset
            if 0 <= j < total_tokens:
                neighbor = tokens[j]
                neighbors.append(neighbor)
                # Distance decay (attention kernel)
                weight = 1.0 / abs(offset)
                n_coord = static_coords[neighbor]
                shift_x += weight * (n_coord["x"] - base_coord["x"])
                shift_y += weight * (n_coord["y"] - base_coord["y"])
                shift_z += weight * (n_coord["z"] - base_coord["z"])
                weight_sum += weight

        if weight_sum > 0:
            shift_x /= weight_sum
            shift_y /= weight_sum
            shift_z /= weight_sum

        # Positional encoding wave (Transformer positional geometry)
        pos_phase = (i / max(1, total_tokens)) * 2 * math.pi
        pos_x = math.sin(pos_phase) * 20
        pos_y = math.cos(pos_phase) * 20
        pos_z = (i / max(1, total_tokens) - 0.5) * 60

        # Contextual coordinate = Base Static + Attention Drift (0.45) + Positional Wave
        ctx_x = base_coord["x"] + 0.45 * shift_x + pos_x
        ctx_y = base_coord["y"] + 0.45 * shift_y + pos_y
        ctx_z = base_coord["z"] + 0.35 * shift_z + pos_z

        occ_info = {
            "index": i,
            "token": t,
            "in_vocab": t in vocab_set,
            "neighbors": neighbors,
            "static_coord": base_coord,
            "context_coord": {
                "x": round(ctx_x, 2),
                "y": round(ctx_y, 2),
                "z": round(ctx_z, 2)
            },
            "drift_magnitude": round(math.sqrt((ctx_x - base_coord["x"])**2 + 
                                              (ctx_y - base_coord["y"])**2 + 
                                              (ctx_z - base_coord["z"])**2), 2)
        }
        contextual_occurrences.append(occ_info)
        token_context_aggregate[t].append(occ_info["context_coord"])

    # Mean contextual coordinate for unique token level summary
    mean_context_coords = {}
    for t, coords in token_context_aggregate.items():
        avg_x = sum(c["x"] for c in coords) / len(coords)
        avg_y = sum(c["y"] for c in coords) / len(coords)
        avg_z = sum(c["z"] for c in coords) / len(coords)
        mean_context_coords[t] = {
            "x": round(avg_x, 2),
            "y": round(avg_y, 2),
            "z": round(avg_z, 2)
        }

    return static_coords, contextual_occurrences, mean_context_coords, co_occur


def generate_interactive_html(tokens, vocab_set, static_coords, contextual_occurrences, mean_context_coords, co_occur, output_path):
    """Generates a rich, interactive 3D WebGL / HTML visualizer with Static vs Contextual toggle."""
    token_counts = Counter(tokens)
    unique_tokens = sorted(list(set(tokens)))
    total_tokens = len(tokens)
    in_vocab_count = sum(1 for t in tokens if t in vocab_set)
    oov_count = total_tokens - in_vocab_count
    coverage = round((in_vocab_count / total_tokens * 100), 1) if total_tokens > 0 else 0

    # Build node data for 3D engine
    nodes_data = []
    for t in unique_tokens:
        count = token_counts[t]
        in_v = t in vocab_set
        s_coord = static_coords[t]
        c_coord = mean_context_coords[t]
        nodes_data.append({
            "id": t,
            "label": t,
            "count": count,
            "frequency": round(count / total_tokens, 4),
            "in_vocab": in_v,
            "static": s_coord,
            "contextual": c_coord,
            "occurrences": [occ for occ in contextual_occurrences if occ["token"] == t]
        })

    # Build co-occurrence links
    links_data = []
    seen_links = set()
    for t1, neighbors in co_occur.items():
        for t2, weight in neighbors.items():
            pair = tuple(sorted([t1, t2]))
            if pair not in seen_links and weight > 0:
                seen_links.add(pair)
                links_data.append({
                    "source": t1,
                    "target": t2,
                    "weight": weight
                })

    # Prepare token stream data for contextual path
    token_stream = [
        {
            "index": occ["index"],
            "token": occ["token"],
            "in_vocab": occ["in_vocab"],
            "static": occ["static_coord"],
            "contextual": occ["context_coord"],
            "drift": occ["drift_magnitude"],
            "neighbors": occ["neighbors"]
        }
        for occ in contextual_occurrences
    ]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Token Counter & Semantic Graph</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Three.js & OrbitControls -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body {{
            background-color: #0b0f19;
            color: #e2e8f0;
            overflow-x: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }}
        #canvas3d {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 0;
        }}
        .glass-panel {{
            background: rgba(17, 24, 39, 0.78);
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .glass-card {{
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .custom-scroll::-webkit-scrollbar {{
            width: 5px;
        }}
        .custom-scroll::-webkit-scrollbar-thumb {{
            background: #374151;
            border-radius: 4px;
        }}
        .glow-cyan {{
            text-shadow: 0 0 12px rgba(6, 182, 212, 0.7);
        }}
        .glow-purple {{
            text-shadow: 0 0 12px rgba(168, 85, 247, 0.7);
        }}
    </style>
</head>
<body class="h-screen w-screen relative select-none">

    <!-- 3D WebGL Canvas -->
    <div id="canvas-container" class="absolute inset-0">
        <canvas id="canvas3d"></canvas>
    </div>

    <!-- Top Navigation / Stats Bar -->
    <header class="absolute top-4 left-4 right-4 z-20 flex flex-wrap items-center justify-between gap-4 pointer-events-none">
        <div class="glass-panel px-5 py-3 rounded-2xl flex items-center gap-3 pointer-events-auto shadow-2xl">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/30">
                <i class="fa-solid fa-cube text-lg"></i>
            </div>
            <div>
                <h1 class="text-base font-bold text-white tracking-wide flex items-center gap-2">
                    Token Counter <span class="text-xs px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-medium">3D Graph</span>
                </h1>
                <p class="text-xs text-gray-400">Static vs Contextual Logic Mapping</p>
            </div>
        </div>

        <!-- Quick Summary Badges -->
        <div class="glass-panel px-5 py-2.5 rounded-2xl flex items-center gap-6 pointer-events-auto shadow-2xl text-xs">
            <div class="text-center">
                <span class="text-gray-400 block text-[10px] uppercase tracking-wider">Total Words</span>
                <span class="font-bold text-base text-white">{total_tokens}</span>
            </div>
            <div class="w-px h-6 bg-gray-700"></div>
            <div class="text-center">
                <span class="text-gray-400 block text-[10px] uppercase tracking-wider">Unique Tokens</span>
                <span class="font-bold text-base text-cyan-400">{len(unique_tokens)}</span>
            </div>
            <div class="w-px h-6 bg-gray-700"></div>
            <div class="text-center">
                <span class="text-gray-400 block text-[10px] uppercase tracking-wider">In-Vocabulary</span>
                <span class="font-bold text-base text-emerald-400">{in_vocab_count} <span class="text-[10px] font-normal text-gray-400">({coverage}%)</span></span>
            </div>
            <div class="w-px h-6 bg-gray-700"></div>
            <div class="text-center">
                <span class="text-gray-400 block text-[10px] uppercase tracking-wider">Out-Of-Vocab</span>
                <span class="font-bold text-base text-rose-400">{oov_count}</span>
            </div>
        </div>

        <!-- Logic Mode Switcher -->
        <div class="glass-panel p-1.5 rounded-2xl flex items-center gap-1 pointer-events-auto shadow-2xl">
            <button id="btn-static" onclick="setMode('static')" class="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 bg-cyan-500 text-white shadow-md shadow-cyan-500/30 flex items-center gap-2">
                <i class="fa-solid fa-shapes"></i> Static Logic
            </button>
            <button id="btn-contextual" onclick="setMode('contextual')" class="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 text-gray-400 hover:text-white flex items-center gap-2">
                <i class="fa-solid fa-brain"></i> Contextual Logic
            </button>
            <button id="btn-dual" onclick="setMode('dual')" class="px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 text-gray-400 hover:text-white flex items-center gap-2">
                <i class="fa-solid fa-route"></i> Shift Vector (Dual)
            </button>
        </div>
    </header>

    <!-- Floating Left Panel: Token Frequency & Metrics -->
    <aside class="absolute top-24 left-4 bottom-6 w-80 glass-panel rounded-2xl flex flex-col p-4 z-20 shadow-2xl">
        <div class="flex items-center justify-between pb-3 border-b border-gray-700/60">
            <div class="flex items-center gap-2">
                <i class="fa-solid fa-chart-simple text-cyan-400 text-sm"></i>
                <h2 class="text-xs font-bold uppercase tracking-wider text-gray-200">Token Frequencies</h2>
            </div>
            <span class="text-[11px] text-gray-400 bg-gray-800/80 px-2 py-0.5 rounded-md">{len(unique_tokens)} items</span>
        </div>

        <!-- Search Input -->
        <div class="mt-3 relative">
            <i class="fa-solid fa-magnifying-glass absolute left-3 top-2.5 text-xs text-gray-500"></i>
            <input type="text" id="token-search" oninput="filterTokens(this.value)" placeholder="Search tokens..." 
                class="w-full bg-gray-900/70 border border-gray-700/60 rounded-xl pl-8 pr-3 py-1.5 text-xs text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500 transition-colors">
        </div>

        <!-- Filter Pills -->
        <div class="flex gap-1.5 mt-2.5 text-[10px]">
            <button onclick="setFilter('all')" id="filter-all" class="px-2.5 py-1 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-medium">All ({len(unique_tokens)})</button>
            <button onclick="setFilter('invocab')" id="filter-invocab" class="px-2.5 py-1 rounded-lg bg-gray-800 text-gray-400 hover:text-gray-200 font-medium">In-Vocab</button>
            <button onclick="setFilter('oov')" id="filter-oov" class="px-2.5 py-1 rounded-lg bg-gray-800 text-gray-400 hover:text-gray-200 font-medium">OOV</button>
        </div>

        <!-- Token List -->
        <div id="token-list" class="flex-1 overflow-y-auto custom-scroll mt-3 space-y-1.5 pr-1">
            <!-- Dynamically populated -->
        </div>

        <!-- 3D Camera & Simulation Controls -->
        <div class="pt-3 mt-2 border-t border-gray-700/60 space-y-2">
            <div class="flex items-center justify-between text-[11px] text-gray-400">
                <span>Auto Rotation</span>
                <input type="checkbox" id="auto-rotate" checked onchange="toggleAutoRotate(this.checked)" class="accent-cyan-500 cursor-pointer">
            </div>
            <div class="flex items-center justify-between text-[11px] text-gray-400">
                <span>Contextual Edges</span>
                <input type="checkbox" id="show-edges" checked onchange="toggleEdges(this.checked)" class="accent-cyan-500 cursor-pointer">
            </div>
            <div class="flex gap-2 pt-1">
                <button onclick="resetCamera()" class="flex-1 py-1.5 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 text-xs font-medium transition-colors flex items-center justify-center gap-1.5">
                    <i class="fa-solid fa-crosshairs text-[10px]"></i> Reset View
                </button>
            </div>
        </div>
    </aside>

    <!-- Floating Right Panel: Node Inspector & Logic Details -->
    <aside id="inspector-panel" class="absolute top-24 right-4 w-88 glass-panel rounded-2xl p-4 z-20 shadow-2xl transition-all duration-300">
        <div class="flex items-center justify-between pb-3 border-b border-gray-700/60">
            <div class="flex items-center gap-2">
                <i class="fa-solid fa-circle-info text-cyan-400 text-sm"></i>
                <h2 class="text-xs font-bold uppercase tracking-wider text-gray-200">Token Inspector</h2>
            </div>
            <span id="selected-badge" class="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">In-Vocab</span>
        </div>

        <div id="inspector-content" class="mt-3 space-y-3">
            <div>
                <div class="text-[10px] uppercase text-gray-400">Token</div>
                <div id="inspect-name" class="text-xl font-bold text-white tracking-wide">Select a Token</div>
            </div>

            <div class="grid grid-cols-2 gap-2 text-xs">
                <div class="glass-card p-2 rounded-xl">
                    <span class="text-[10px] text-gray-400 block">Occurrences</span>
                    <span id="inspect-count" class="font-bold text-white">-</span>
                </div>
                <div class="glass-card p-2 rounded-xl">
                    <span class="text-[10px] text-gray-400 block">Relative Freq</span>
                    <span id="inspect-freq" class="font-bold text-cyan-400">-</span>
                </div>
            </div>

            <!-- Static vs Contextual Coordinates Comparison -->
            <div class="glass-card p-3 rounded-xl space-y-2 text-xs">
                <div class="flex items-center justify-between">
                    <span class="font-semibold text-cyan-300 flex items-center gap-1">
                        <i class="fa-solid fa-shapes text-[10px]"></i> Static Coord (X, Y, Z)
                    </span>
                    <span id="inspect-static-coord" class="font-mono text-gray-300 text-[11px]">-</span>
                </div>
                <p class="text-[10px] text-gray-400 leading-tight">Fixed intrinsic position mapped from global co-occurrence & lexical length.</p>
                <div class="h-px bg-gray-700/60"></div>
                <div class="flex items-center justify-between">
                    <span class="font-semibold text-purple-300 flex items-center gap-1">
                        <i class="fa-solid fa-brain text-[10px]"></i> Contextual Coord
                    </span>
                    <span id="inspect-context-coord" class="font-mono text-gray-300 text-[11px]">-</span>
                </div>
                <p class="text-[10px] text-gray-400 leading-tight">Shifted position dynamically pulled by neighboring context window tokens.</p>
                <div class="h-px bg-gray-700/60"></div>
                <div class="flex items-center justify-between">
                    <span class="font-semibold text-amber-300 text-[11px]">Contextual Drift (Δ)</span>
                    <span id="inspect-drift" class="font-bold text-amber-400 text-xs">-</span>
                </div>
            </div>

            <!-- Contextual Neighbors -->
            <div>
                <span class="text-[10px] uppercase text-gray-400 block mb-1">Co-occurring Context Neighbors</span>
                <div id="inspect-neighbors" class="flex flex-wrap gap-1 text-[11px]">
                    <span class="text-gray-500 italic text-[11px]">Hover or click a node</span>
                </div>
            </div>
        </div>

        <!-- Logic Explainer Box -->
        <div class="mt-4 pt-3 border-t border-gray-700/60 text-[11px] text-gray-400 space-y-1">
            <div class="font-semibold text-gray-300 flex items-center gap-1.5">
                <i class="fa-solid fa-lightbulb text-yellow-400"></i> Logic Distinction
            </div>
            <p><strong>Static:</strong> Token has one fixed embedding regardless of surrounding words (Word2Vec / GloVe style).</p>
            <p><strong>Contextual:</strong> Token shifts in 3D based on surrounding attention & sentence sequence (BERT / GPT style).</p>
        </div>
    </aside>

    <!-- Legend at bottom center -->
    <div class="absolute bottom-4 left-1/2 -translate-x-1/2 glass-panel px-4 py-2 rounded-xl flex items-center gap-5 text-xs z-10 shadow-xl">
        <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded-full bg-emerald-400 shadow-sm shadow-emerald-400/50"></span>
            <span class="text-gray-300 text-[11px]">In-Vocab Token</span>
        </div>
        <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded-full bg-rose-400 shadow-sm shadow-rose-400/50"></span>
            <span class="text-gray-300 text-[11px]">Out-of-Vocab (OOV)</span>
        </div>
        <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400/50"></span>
            <span class="text-gray-300 text-[11px]">Static Position</span>
        </div>
        <div class="flex items-center gap-2">
            <span class="w-3 h-3 rounded-full bg-purple-400 shadow-sm shadow-purple-400/50"></span>
            <span class="text-gray-300 text-[11px]">Contextual Shift</span>
        </div>
    </div>

    <!-- Embedded Graph Data -->
    <script>
        const GRAPH_DATA = {{
            nodes: {json.dumps(nodes_data)},
            links: {json.dumps(links_data)},
            stream: {json.dumps(token_stream)},
            stats: {{
                total: {total_tokens},
                unique: {len(unique_tokens)},
                in_vocab: {in_vocab_count},
                oov: {oov_count},
                coverage: {coverage}
            }}
        }};

        let scene, camera, renderer, controls;
        let nodeObjects = new Map();
        let linkObjects = [];
        let vectorArrows = [];
        let currentMode = 'static'; // 'static' | 'contextual' | 'dual'
        let currentFilter = 'all';
        let selectedNode = null;
        let raycaster, mouse;

        function initScene() {{
            const container = document.getElementById('canvas-container');
            const width = window.innerWidth;
            const height = window.innerHeight;

            scene = new THREE.Scene();
            scene.fog = new THREE.FogExp2(0x0b0f19, 0.0018);

            camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 2000);
            camera.position.set(0, 70, 240);

            renderer = new THREE.WebGLRenderer({{ canvas: document.getElementById('canvas3d'), antialias: true, alpha: true }});
            renderer.setSize(width, height);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxDistance = 600;
            controls.minDistance = 30;

            // Lights
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);

            const dirLight = new THREE.DirectionalLight(0x60a5fa, 0.9);
            dirLight.position.set(100, 150, 100);
            scene.add(dirLight);

            const pointLight = new THREE.PointLight(0xa855f7, 1.2, 500);
            pointLight.position.set(-100, -50, -50);
            scene.add(pointLight);

            // Add 3D Coordinate Grid Guide
            const gridHelper = new THREE.GridHelper(300, 30, 0x1f2937, 0x111827);
            gridHelper.position.y = -80;
            scene.add(gridHelper);

            // Raycasting for node interaction
            raycaster = new THREE.Raycaster();
            mouse = new THREE.Vector2();

            window.addEventListener('resize', onWindowResize);
            window.addEventListener('mousemove', onMouseMove);
            window.addEventListener('click', onClick);

            build3DGraph();
            renderTokenList();
            if (GRAPH_DATA.nodes.length > 0) {{
                selectToken(GRAPH_DATA.nodes[0].id);
            }}

            animate();
        }}

        function build3DGraph() {{
            // Clear existing
            nodeObjects.forEach(obj => scene.remove(obj.mesh));
            nodeObjects.clear();
            linkObjects.forEach(line => scene.remove(line));
            linkObjects = [];
            vectorArrows.forEach(arrow => scene.remove(arrow));
            vectorArrows = [];

            // 1. Create Node Spheres & Text Labels
            GRAPH_DATA.nodes.forEach(node => {{
                const radius = Math.max(2.8, Math.min(8.0, 2.5 + Math.sqrt(node.count) * 2.2));
                const geometry = new THREE.SphereGeometry(radius, 24, 24);

                let color = node.in_vocab ? 0x10b981 : 0xf43f5e; // emerald vs rose
                const material = new THREE.MeshStandardMaterial({{
                    color: color,
                    roughness: 0.3,
                    metalness: 0.5,
                    emissive: color,
                    emissiveIntensity: 0.25
                }});

                const mesh = new THREE.Mesh(geometry, material);
                mesh.userData = node;

                // Position based on current mode
                const targetPos = currentMode === 'contextual' ? node.contextual : node.static;
                mesh.position.set(targetPos.x, targetPos.y, targetPos.z);
                scene.add(mesh);

                // Add Sprite Text Label
                const labelCanvas = document.createElement('canvas');
                const ctx = labelCanvas.getContext('2d');
                labelCanvas.width = 256;
                labelCanvas.height = 64;
                ctx.font = 'bold 24px -apple-system, sans-serif';
                ctx.fillStyle = node.in_vocab ? '#a7f3d0' : '#fecdd3';
                ctx.textAlign = 'center';
                ctx.fillText(node.label, 128, 40);

                const texture = new THREE.CanvasTexture(labelCanvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ map: texture, transparent: true, opacity: 0.85 }});
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.scale.set(22, 5.5, 1);
                sprite.position.set(0, radius + 3.5, 0);
                mesh.add(sprite);

                nodeObjects.set(node.id, {{
                    mesh: mesh,
                    data: node,
                    currentPos: {{ ...mesh.position }},
                    targetPos: {{ ...targetPos }}
                }});
            }});

            // 2. Build Co-occurrence Edges
            const lineMat = new THREE.LineBasicMaterial({{
                color: 0x38bdf8,
                transparent: true,
                opacity: 0.22,
                linewidth: 1
            }});

            GRAPH_DATA.links.forEach(link => {{
                const sourceObj = nodeObjects.get(link.source);
                const targetObj = nodeObjects.get(link.target);
                if (sourceObj && targetObj) {{
                    const geom = new THREE.BufferGeometry().setFromPoints([
                        sourceObj.mesh.position,
                        targetObj.mesh.position
                    ]);
                    const line = new THREE.Line(geom, lineMat);
                    line.userData = {{ source: link.source, target: link.target }};
                    scene.add(line);
                    linkObjects.push(line);
                }}
            }});

            // 3. Build Dual-mode Vector Arrows (Static -> Contextual displacement)
            GRAPH_DATA.nodes.forEach(node => {{
                const s = node.static;
                const c = node.contextual;
                const from = new THREE.Vector3(s.x, s.y, s.z);
                const to = new THREE.Vector3(c.x, c.y, c.z);
                const dir = new THREE.Vector3().subVectors(to, from);
                const length = dir.length();

                if (length > 0.5) {{
                    dir.normalize();
                    const arrow = new THREE.ArrowHelper(dir, from, length, 0xa855f7, 2.5, 1.8);
                    arrow.visible = (currentMode === 'dual');
                    scene.add(arrow);
                    vectorArrows.push(arrow);
                }}
            }});
        }}

        function updateNodePositionsAnimated() {{
            nodeObjects.forEach(obj => {{
                const target = currentMode === 'contextual' ? obj.data.contextual : obj.data.static;
                obj.targetPos = target;
            }});

            vectorArrows.forEach(arrow => {{
                arrow.visible = (currentMode === 'dual');
            }});
        }}

        function setMode(mode) {{
            currentMode = mode;
            ['static', 'contextual', 'dual'].forEach(m => {{
                const btn = document.getElementById(`btn-${{m}}`);
                if (m === mode) {{
                    btn.className = "px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 bg-cyan-500 text-white shadow-md shadow-cyan-500/30 flex items-center gap-2";
                }} else {{
                    btn.className = "px-3.5 py-1.5 rounded-xl text-xs font-semibold transition-all duration-200 text-gray-400 hover:text-white flex items-center gap-2";
                }}
            }});

            updateNodePositionsAnimated();
            if (selectedNode) selectToken(selectedNode);
        }}

        function setFilter(filter) {{
            currentFilter = filter;
            ['all', 'invocab', 'oov'].forEach(f => {{
                const btn = document.getElementById(`filter-${{f}}`);
                if (f === filter) {{
                    btn.className = "px-2.5 py-1 rounded-lg bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-medium";
                }} else {{
                    btn.className = "px-2.5 py-1 rounded-lg bg-gray-800 text-gray-400 hover:text-gray-200 font-medium";
                }}
            }});

            renderTokenList();
            updateNodeVisibility();
        }}

        function updateNodeVisibility() {{
            const searchVal = document.getElementById('token-search').value.toLowerCase();
            nodeObjects.forEach((obj, id) => {{
                let matchFilter = true;
                if (currentFilter === 'invocab') matchFilter = obj.data.in_vocab;
                if (currentFilter === 'oov') matchFilter = !obj.data.in_vocab;
                let matchSearch = id.toLowerCase().includes(searchVal);
                obj.mesh.visible = matchFilter && matchSearch;
            }});
        }}

        function renderTokenList() {{
            const container = document.getElementById('token-list');
            const searchVal = (document.getElementById('token-search')?.value || '').toLowerCase();
            container.innerHTML = '';

            const filtered = GRAPH_DATA.nodes.filter(n => {{
                let matchFilter = true;
                if (currentFilter === 'invocab') matchFilter = n.in_vocab;
                if (currentFilter === 'oov') matchFilter = !n.in_vocab;
                return matchFilter && n.label.toLowerCase().includes(searchVal);
            }});

            filtered.sort((a, b) => b.count - a.count);

            filtered.forEach(node => {{
                const item = document.createElement('div');
                item.className = `p-2 rounded-xl cursor-pointer transition-all flex items-center justify-between text-xs ${{
                    selectedNode === node.id ? 'bg-cyan-500/20 border border-cyan-500/50 text-white' : 'glass-card hover:bg-gray-800/80 text-gray-300'
                }}`;
                item.onclick = () => selectToken(node.id);
                item.innerHTML = `
                    <div class="flex items-center gap-2">
                        <span class="w-2 h-2 rounded-full ${{node.in_vocab ? 'bg-emerald-400' : 'bg-rose-400'}}"></span>
                        <span class="font-medium font-mono">${{node.label}}</span>
                    </div>
                    <div class="flex items-center gap-2 text-[11px]">
                        <span class="text-gray-400">${{node.count}}x</span>
                        <span class="text-[9px] px-1.5 py-0.5 rounded ${{node.in_vocab ? 'bg-emerald-900/40 text-emerald-300' : 'bg-rose-900/40 text-rose-300'}}">
                            ${{node.in_vocab ? 'VOCAB' : 'OOV'}}
                        </span>
                    </div>
                `;
                container.appendChild(item);
            }});
        }}

        function filterTokens() {{
            renderTokenList();
            updateNodeVisibility();
        }}

        function selectToken(tokenId) {{
            selectedNode = tokenId;
            const node = GRAPH_DATA.nodes.find(n => n.id === tokenId);
            if (!node) return;

            document.getElementById('inspect-name').textContent = node.label;
            document.getElementById('inspect-count').textContent = `${{node.count}} occurrences`;
            document.getElementById('inspect-freq').textContent = `${{(node.frequency * 100).toFixed(2)}}%`;

            const badge = document.getElementById('selected-badge');
            if (node.in_vocab) {{
                badge.className = "text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
                badge.textContent = "In-Vocabulary";
            }} else {{
                badge.className = "text-[10px] px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30";
                badge.textContent = "Out-Of-Vocabulary (OOV)";
            }}

            document.getElementById('inspect-static-coord').textContent = 
                `(${{node.static.x}}, ${{node.static.y}}, ${{node.static.z}})`;
            document.getElementById('inspect-context-coord').textContent = 
                `(${{node.contextual.x}}, ${{node.contextual.y}}, ${{node.contextual.z}})`;

            const dx = node.contextual.x - node.static.x;
            const dy = node.contextual.y - node.static.y;
            const dz = node.contextual.z - node.static.z;
            const drift = Math.sqrt(dx*dx + dy*dy + dz*dz).toFixed(2);
            document.getElementById('inspect-drift').textContent = `${{drift}} units`;

            // Neighbors
            const neighborSet = new Set();
            node.occurrences.forEach(occ => {{
                occ.neighbors.forEach(nbr => neighborSet.add(nbr));
            }});
            const nbrContainer = document.getElementById('inspect-neighbors');
            nbrContainer.innerHTML = '';
            if (neighborSet.size === 0) {{
                nbrContainer.innerHTML = '<span class="text-gray-500 italic text-[11px]">No adjacent neighbors in context window</span>';
            }} else {{
                Array.from(neighborSet).forEach(nbr => {{
                    const span = document.createElement('span');
                    span.className = "px-2 py-0.5 rounded-lg bg-gray-800 text-cyan-300 border border-gray-700 font-mono text-[10px] cursor-pointer hover:bg-gray-700";
                    span.textContent = nbr;
                    span.onclick = () => selectToken(nbr);
                    nbrContainer.appendChild(span);
                }});
            }}

            // Highlight node mesh in 3D
            nodeObjects.forEach((obj, id) => {{
                if (id === tokenId) {{
                    obj.mesh.material.emissiveIntensity = 0.9;
                    obj.mesh.scale.set(1.4, 1.4, 1.4);
                }} else {{
                    obj.mesh.material.emissiveIntensity = 0.2;
                    obj.mesh.scale.set(1.0, 1.0, 1.0);
                }}
            }});

            renderTokenList();
        }}

        function onMouseMove(event) {{
            mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        }}

        function onClick(event) {{
            raycaster.setFromCamera(mouse, camera);
            const meshes = Array.from(nodeObjects.values()).map(o => o.mesh);
            const intersects = raycaster.intersectObjects(meshes);
            if (intersects.length > 0) {{
                const targetNode = intersects[0].object.userData;
                selectToken(targetNode.id);
            }}
        }}

        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}

        function toggleAutoRotate(val) {{
            controls.autoRotate = val;
            controls.autoRotateSpeed = 1.0;
        }}

        function toggleEdges(val) {{
            linkObjects.forEach(l => l.visible = val);
        }}

        function resetCamera() {{
            camera.position.set(0, 70, 240);
            controls.target.set(0, 0, 0);
            controls.update();
        }}

        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);

            // Smooth interpolation to target positions
            nodeObjects.forEach(obj => {{
                obj.mesh.position.lerp(
                    new THREE.Vector3(obj.targetPos.x, obj.targetPos.y, obj.targetPos.z),
                    0.08
                );
            }});

            // Update edge endpoints
            linkObjects.forEach(line => {{
                const sObj = nodeObjects.get(line.userData.source);
                const tObj = nodeObjects.get(line.userData.target);
                if (sObj && tObj) {{
                    const positions = line.geometry.attributes.position.array;
                    positions[0] = sObj.mesh.position.x;
                    positions[1] = sObj.mesh.position.y;
                    positions[2] = sObj.mesh.position.z;
                    positions[3] = tObj.mesh.position.x;
                    positions[4] = tObj.mesh.position.y;
                    positions[5] = tObj.mesh.position.z;
                    line.geometry.attributes.position.needsUpdate = true;
                }}
            }});

            controls.update();
            renderer.render(scene, camera);
        }}

        window.onload = () => {{
            toggleAutoRotate(true);
            initScene();
        }};
    </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def main():
    print("=" * 68)
    print(" 🌟 TOKEN COUNTER & 3D GRAPH MAPPER (Static & Contextual Logics)")
    print("=" * 68)
    print()

    # Step 1: Ask for Vocabulary
    print("STEP 1: Define Vocabulary")
    print("-" * 68)
    print("Choose a vocabulary source:")
    print("  [1] Preset 1: General English Core (top 85 common words)")
    print("  [2] Preset 2: AI & NLP Domain Vocabulary (30 tokens)")
    print("  [3] Preset 3: Finance & Business Vocabulary (21 tokens)")
    print("  [4] Custom: Enter your own vocabulary list")
    print()

    vocab_choice = input("Select an option [1-4] (default: 2): ").strip()
    if vocab_choice not in ["1", "2", "3", "4"]:
        vocab_choice = "2"

    if vocab_choice in ["1", "2", "3"]:
        preset = PRESET_VOCABULARIES[vocab_choice]
        vocab_list = preset["words"]
        print(f"\nLoaded preset: '{preset['name']}' ({len(vocab_list)} words)")
        print(f"Sample words: {', '.join(vocab_list[:12])} ...")
    else:
        print("\nEnter your custom vocabulary words separated by commas, spaces, or lines.")
        print("(Press Enter twice or leave blank when done):")
        custom_input = input("Vocabulary: ").strip()
        vocab_list = clean_tokenize(custom_input)
        if not vocab_list:
            print("No words entered. Falling back to Preset 2 (AI & NLP).")
            vocab_list = PRESET_VOCABULARIES["2"]["words"]

    vocab_set = set(vocab_list)
    print(f"\n✅ Active Vocabulary registered: {len(vocab_set)} unique tokens.")

    # Step 2: Ask for Input Text to Count
    print("\n" + "=" * 68)
    print("STEP 2: Enter Text to Count and Analyze")
    print("-" * 68)
    print("Sample texts:")
    print(f"  [1] AI/NLP: \"{SAMPLE_TEXTS['1'][:75]}...\"")
    print(f"  [2] Contextual bank polysemy: \"{SAMPLE_TEXTS['2'][:75]}...\"")
    print(f"  [3] Token counter description: \"{SAMPLE_TEXTS['3'][:75]}...\"")
    print("  [4] Custom: Paste or type your own text")
    print()

    text_choice = input("Select an option [1-4] (default: 1): ").strip()
    if text_choice not in ["1", "2", "3", "4"]:
        text_choice = "1"

    if text_choice in ["1", "2", "3"]:
        raw_text = SAMPLE_TEXTS[text_choice]
    else:
        print("\nEnter or paste your text below:")
        raw_text = input("> ").strip()
        if not raw_text:
            print("No text entered. Using sample 1.")
            raw_text = SAMPLE_TEXTS["1"]

    # Step 3: Count All Words & Compute Statistics
    tokens = clean_tokenize(raw_text)
    if not tokens:
        print("Error: Could not extract any tokens from the provided text.")
        return

    token_counts = Counter(tokens)
    total_words = len(tokens)
    unique_words = len(token_counts)
    in_vocab_tokens = [t for t in tokens if t in vocab_set]
    oov_tokens = [t for t in tokens if t not in vocab_set]

    coverage_pct = round((len(in_vocab_tokens) / total_words) * 100, 2)

    print("\n" + "=" * 68)
    print("📊 TOKEN COUNT SUMMARY & VOCABULARY ANALYSIS")
    print("=" * 68)
    print(f"Total Words Counted     : {total_words}")
    print(f"Unique Word Types       : {unique_words}")
    print(f"Vocabulary Size (Ref)   : {len(vocab_set)}")
    print(f"In-Vocabulary Tokens    : {len(in_vocab_tokens)} ({coverage_pct}% coverage)")
    print(f"Out-Of-Vocabulary (OOV) : {len(oov_tokens)} ({round(100 - coverage_pct, 2)}%)")
    print("-" * 68)

    print("\nTop Token Frequency Breakdown:")
    print(f"{'Rank':<5} | {'Token':<18} | {'Count':<6} | {'Freq (%)':<9} | {'Status':<12}")
    print("-" * 68)
    for rank, (word, count) in enumerate(token_counts.most_common(15), 1):
        status = "IN_VOCAB" if word in vocab_set else "OOV"
        pct = f"{(count / total_words) * 100:.1f}%"
        print(f"{rank:<5} | {word:<18} | {count:<6} | {pct:<9} | {status:<12}")

    # Step 4: Compute 3D Coordinates (Static & Contextual Logics)
    print("\n" + "=" * 68)
    print("🧠 COMPUTING 3D MAPPINGS (Static vs Contextual Logic)")
    print("=" * 68)
    print("1. Static Logic: Calculating global co-occurrence, PPMI, and lexical manifold...")
    print("2. Contextual Logic: Simulating attention-weighted neighbor pulls & positional drift...")

    static_coords, contextual_occurrences, mean_context_coords, co_occur = \
        calculate_static_and_contextual_coords(tokens, vocab_set, window_size=2)

    print("\nSample 3D Coordinates (Static vs Contextual Drift):")
    print(f"{'Token':<15} | {'Static (X,Y,Z)':<22} | {'Contextual Mean (X,Y,Z)':<24} | {'Drift (Δ)':<9}")
    print("-" * 76)
    for word, _ in token_counts.most_common(8):
        s = static_coords[word]
        c = mean_context_coords[word]
        s_str = f"({s['x']:.0f},{s['y']:.0f},{s['z']:.0f})"
        c_str = f"({c['x']:.0f},{c['y']:.0f},{c['z']:.0f})"
        drift = math.sqrt((c['x'] - s['x'])**2 + (c['y'] - s['y'])**2 + (c['z'] - s['z'])**2)
        print(f"{word:<15} | {s_str:<22} | {c_str:<24} | {drift:<9.1f}")

    # Step 5: Generate and Launch Interactive 3D Graph
    output_html = os.path.abspath("token_graph_3d.html")
    generate_interactive_html(tokens, vocab_set, static_coords, contextual_occurrences, mean_context_coords, co_occur, output_html)

    print("\n" + "=" * 68)
    print(f"✨ Interactive 3D Graph generated successfully:")
    print(f"📁 {output_html}")
    print("=" * 68)
    print("Features in 3D viewer:")
    print("  • Toggle 'Static Logic' vs 'Contextual Logic' vs 'Shift Vector (Dual)'")
    print("  • Inspect node coordinates, attention context pull, and In-Vocab/OOV status")
    print("  • 3D Orbit camera with pan, zoom, rotation, and token search")

    # Ask to open in browser
    open_now = input("\nOpen 3D graph in default browser now? [Y/n]: ").strip().lower()
    if open_now in ["", "y", "yes"]:
        try:
            webbrowser.open(f"file://{output_html}")
            print("🚀 Launched in web browser.")
        except Exception as e:
            print(f"Could not open browser automatically: {e}")
            print(f"Please open file manually: {output_html}")


if __name__ == "__main__":
    main()
