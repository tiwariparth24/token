# 🌟 Token Counter & 3D Semantic Graph Mapper

A token counter and 3D spatial representation engine that asks for a vocabulary, counts words/tokens from input text, checks in-vocab vs out-of-vocab (OOV) tokens, and maps all tokens into an interactive **3D graph** using **Static** and **Contextual** logic.

---

## 🚀 Quick Start

You can run this project in two ways:

### 1. Interactive Web Application (Recommended)
Simply open `index.html` in your browser:
```bash
open index.html
```
*Or run the local web server:*
```bash
python3 server.py
```
Then visit `http://localhost:8000/index.html`.

### 2. Interactive Terminal CLI
Run the standalone Python script (zero dependencies, works on pure Python 3):
```bash
python3 token_counter.py
```
This CLI will prompt you for:
1. **Vocabulary:** Choose a preset (AI & NLP, Common English, Finance) or enter your own custom words.
2. **Text Input:** Choose a sample text or enter your own passage.
3. **Counts:** Displays formatted tables of frequencies, in-vocab matches, and OOV rates.
4. **3D Graph:** Generates an interactive 3D WebGL visualization (`token_graph_3d.html`) and launches it in your browser.

---

## 🧠 Core Architecture & Logics

### 1. Vocabulary Phase
- Asks user for reference vocabulary words (comma, space, or newline separated).
- Normalizes tokens and stores them in an active vocabulary dictionary.
- All subsequent text is analyzed against this reference vocabulary.

### 2. Token Counting Phase
- Tokenizes incoming text stream into lexical units.
- Calculates:
  - **Total Words**: Total occurrences in corpus.
  - **Unique Tokens**: Number of distinct word types.
  - **In-Vocabulary Tokens**: Words recognized in the vocabulary with coverage rate percentage.
  - **Out-of-Vocabulary (OOV) Tokens**: New, unmapped tokens.
  - **Type-Token Ratio (TTR)**: Lexical diversity metric.
  - **Word Frequency Table**: Rank, raw counts, relative frequency %, and status.

### 3. 3D Spatial Representation Logics

| Logic | Concept | Analogy in NLP | Math / Behavior |
| :--- | :--- | :--- | :--- |
| **Static Logic** | Global corpus & lexical coordinates | Word2Vec, GloVe, FastText | Tokens have **fixed coordinates** determined by frequency, global co-occurrence graph centrality, and token length. Every instance of a word shares the exact same static coordinate. |
| **Contextual Logic** | Attention & sequence-shifted coordinates | Transformers, BERT, GPT | Each token's position is **dynamically pulled** by the coordinates of surrounding words in the context window ($W$) plus positional encoding. Identical words in different contexts occupy distinct spatial locations! |
| **Shift Vector (Dual)** | Contextual drift visualization | Vector displacement ($\Delta \vec{v}$) | Shows purple directional vectors indicating how context pulls each token from its base static position to its contextual representation. |

---

## 🛠️ File Structure

- `index.html`: Fully responsive, dark-mode WebGL application featuring real-time vocabulary management, token counting, 3D OrbitControls, node inspector, and dual-logic toggling.
- `token_counter.py`: Terminal CLI tool for vocabulary definition, word counting, and 3D HTML export without any external pip requirements.
- `server.py`: Lightweight Python server to serve the web app locally.
- `README.md`: System documentation and mathematical guide.

---

## 🎮 3D Graph Controls
- **Rotate**: Left Click + Drag
- **Pan**: Right Click + Drag (or Shift + Drag)
- **Zoom**: Scroll Wheel
- **Inspect Node**: Click or hover any sphere to inspect its static $(X,Y,Z)$, contextual $(X,Y,Z)$, drift distance, frequency, and local neighbors.
- **Mode Toggle**: Switch between **Static Logic**, **Contextual Logic**, and **Shift Vector (Dual)** on the fly.
