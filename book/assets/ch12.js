/* Chapter 12 — concept map, full timeline, searchable glossary.
   Timeline = shared MI_TIMELINE (assets/timeline.js) + page-only EXTRA entries below.
   Glossary = shared MI_GLOSSARY (assets/glossary.js) only. */
(function () {
  const CH = n => (window.MI_CHAPTERS || []).find(c => c.n === n);
  const chLink = n => { const c = CH(n); return c ? `<a href="${c.file}">Ch. ${n} · ${c.title}</a>` : ""; };

  /* =========================================================
     1. Concept map
     ========================================================= */
  const NODES = [
    // col 1 — roots (2013–2019)
    { id: "saliency", label: "Saliency & attribution", x: 95, y: 60, ch: 2, d: "Gradient maps that show which input pixels or tokens mattered. Useful, but they say where, not how." },
    { id: "featvis", label: "Feature visualization", x: 95, y: 150, ch: 2, d: "Optimize an input until a neuron fires strongly, to see what the neuron detects." },
    { id: "netdiss", label: "Dissection & TCAV", x: 95, y: 240, ch: 2, d: "Score units against labelled concepts (Network Dissection), or find concept directions (TCAV)." },
    { id: "probing", label: "Probing", x: 95, y: 330, ch: 7, d: "Train a small classifier on activations to test if some information is present there." },
    { id: "resid", label: "Residual stream", x: 95, y: 450, ch: 4, d: "The shared vector that every transformer layer reads from and adds to. A communication channel." },
    { id: "ablation", label: "Ablation", x: 95, y: 540, ch: 5, d: "Remove or replace a component (zero, mean, or resample) and measure what changes." },
    // col 2 — circuits era (2020–2021)
    { id: "universality", label: "Universality", x: 265, y: 60, ch: 2, d: "Similar features and circuits appear in different models trained on similar data." },
    { id: "circuits", label: "Circuits (vision)", x: 265, y: 150, ch: 2, d: "Features connected by weights that compute new features, e.g. curve detectors in InceptionV1." },
    { id: "polysem", label: "Polysemanticity", x: 265, y: 240, ch: 3, d: "Many neurons respond to several unrelated things, such as cat faces and car fronts." },
    { id: "lrh", label: "Linear representations", x: 265, y: 330, ch: 3, d: "The hypothesis that concepts are directions in activation space, readable with a dot product." },
    { id: "qkov", label: "Attention: QK & OV", x: 265, y: 450, ch: 4, d: "Each head has a QK circuit (where to look) and an OV circuit (what to move)." },
    { id: "patching", label: "Activation patching", x: 265, y: 540, ch: 5, d: "Copy an activation from a clean run into a corrupted run and see how much behaviour comes back." },
    // col 3 — LM circuits & superposition (2022–2023)
    { id: "superposition", label: "Superposition", x: 435, y: 240, ch: 3, d: "Networks store more features than dimensions, in almost-orthogonal directions. Needs sparsity." },
    { id: "steering", label: "Directions & steering", x: 435, y: 330, ch: 7, d: "A concept is often a direction, found with a probe or a simple difference of means. Read it to monitor, or add it at run time to steer, e.g. the refusal direction." },
    { id: "induction", label: "Induction heads", x: 435, y: 450, ch: 4, d: "A two-head circuit that continues repeated patterns: [A][B] … [A] → [B]. Drives in-context learning." },
    { id: "pathpatch", label: "Path patching", x: 435, y: 540, ch: 5, d: "Patch only the signal along one path between two components, to map edges of a circuit." },
    // col 4
    { id: "autointerp", label: "Auto-interpretability", x: 605, y: 150, ch: 8, d: "A language model writes and scores explanations of neurons or features at scale." },
    { id: "sae", label: "Sparse autoencoders", x: 605, y: 240, ch: 8, d: "Dictionary learning: rewrite each activation as a sparse sum of many learned feature directions." },
    { id: "lmcircuits", label: "LM circuits (IOI…)", x: 605, y: 360, ch: 6, d: "Hand-found algorithms in language models: IOI, greater-than, modular addition, Othello world model." },
    { id: "das", label: "DAS & abstraction", x: 605, y: 450, ch: 5, d: "Test whether a network implements a high-level algorithm by aligning variables with subspaces." },
    { id: "scrubbing", label: "Scrubbing & ACDC", x: 605, y: 540, ch: 5, d: "Causal scrubbing tests a circuit hypothesis; ACDC finds circuits automatically by pruning edges." },
    // col 5
    { id: "diffing", label: "Crosscoders & diffing", x: 775, y: 150, ch: 9, d: "Dictionaries shared across layers or models. Compare a base model and its chat version." },
    { id: "transcoders", label: "Transcoders & CLTs", x: 775, y: 240, ch: 9, d: "Sparse replacements for MLPs. Their features connect cleanly, so circuits can be traced." },
    { id: "paramdecomp", label: "Weight decomposition", x: 775, y: 360, ch: 9, d: "Decompose the weights themselves into simple mechanisms (APD, SPD), not the activations." },
    { id: "devinterp", label: "Developmental interp", x: 775, y: 450, ch: 10, d: "Study how structure forms during training, e.g. with the local learning coefficient." },
    // col 6 — 2025
    { id: "auditing", label: "Auditing games & agents", x: 945, y: 60, ch: 11, d: "Hide a goal in a model organism, then see whether teams or AI agents with interpretability tools can find it (2025)." },
    { id: "selfexplain", label: "Self-explaining models", x: 945, y: 150, ch: 10, d: "Prompt or train a model to describe activations in words: Patchscopes (2024), LatentQA, self-explainers, Activation Oracles and PCDs (2025)." },
    { id: "attrgraph", label: "Attribution graphs", x: 945, y: 240, ch: 9, d: "Per-prompt graphs of feature-to-feature effects in a local replacement model, checked with interventions in the real model." },
    // col 7 — 2026
    { id: "sysaudit", label: "White-box audits", x: 1115, y: 60, ch: 11, d: "Since September 2025, every major Anthropic system card has a white-box audit: SAE features, attribution graphs and NLAs look for evaluation awareness and hidden reasoning." },
    { id: "nla", label: "NL autoencoders", x: 1115, y: 150, ch: 10, d: "Natural language autoencoders (2026): one model copy turns an activation into text, another rebuilds the activation from it. Critics test whether such explainers really read the activation." },
    { id: "prodprobes", label: "Probes in production", x: 1115, y: 330, ch: 11, d: "Cheap linear probes on live traffic: Gemini misuse probes and Constitutional Classifiers++ (January 2026). Models can be trained to evade probes (Neural Chameleons)." },
    { id: "personas", label: "Personas in the model", x: 1115, y: 450, ch: 10, d: "Persona vectors, then the whole space of characters: one Assistant Axis measures how much the model acts as its default Assistant (2026)." },
  ];
  const EDGES = [
    ["featvis", "circuits", "b"], ["circuits", "universality", "b"], ["saliency", "featvis", "m"],
    ["featvis", "polysem", "m"], ["netdiss", "probing", "b"], ["polysem", "superposition", "m"],
    ["lrh", "superposition", "b"], ["probing", "lrh", "t"], ["lrh", "steering", "b"], ["probing", "steering", "b"],
    ["superposition", "sae", "m"], ["resid", "qkov", "b"], ["qkov", "induction", "b"], ["induction", "lmcircuits", "b"],
    ["circuits", "lmcircuits", "b"], ["ablation", "patching", "b"], ["patching", "pathpatch", "b"],
    ["pathpatch", "lmcircuits", "t"], ["pathpatch", "scrubbing", "b"], ["patching", "das", "b"], ["das", "lmcircuits", "t"],
    ["scrubbing", "lmcircuits", "t"], ["sae", "autointerp", "b"], ["sae", "transcoders", "b"], ["sae", "diffing", "b"],
    ["transcoders", "attrgraph", "b"], ["lmcircuits", "attrgraph", "m"], ["diffing", "auditing", "b"],
    ["attrgraph", "auditing", "b"], ["autointerp", "selfexplain", "b"], ["sae", "steering", "b"],
    ["steering", "auditing", "b"], ["induction", "devinterp", "m"], ["superposition", "paramdecomp", "m"],
    ["patching", "attrgraph", "t"], ["auditing", "sysaudit", "b"], ["selfexplain", "nla", "b"], ["nla", "sysaudit", "b"],
    ["attrgraph", "sysaudit", "b"], ["steering", "prodprobes", "b"], ["steering", "personas", "b"],
  ];
  const NW = 152, NH = 36;
  const svg = document.getElementById("cmap");
  const info = document.getElementById("cmapInfo");
  if (svg) {
    const byId = Object.fromEntries(NODES.map(n => [n.id, n]));
    const kindStyle = { b: { c: "var(--ink-3)", dash: "" }, m: { c: "var(--accent)", dash: "6 4" }, t: { c: "var(--blue)", dash: "" } };
    const defs = MI.svg("defs", {}, svg);
    Object.entries(kindStyle).forEach(([k, s]) => {
      const m = MI.svg("marker", { id: "cm-" + k, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
      MI.svg("path", { d: "M0,0 L10,5 L0,10 z", fill: s.c }, m);
    });
    // era columns
    const eras = [["Roots 2013–19", 95], ["Circuits 2020–21", 265], ["LM circuits 2022–23", 435], ["Scaling 2023–24", 605], ["Replacement 2024–25", 775], ["2025", 945], ["2026", 1115]];
    eras.forEach(([t, x], i) => {
      MI.svg("rect", { x: x - 84, y: 8, width: 168, height: 584, rx: 10, fill: i % 2 ? "var(--bg-2)" : "none", opacity: 0.7 }, svg);
      MI.svg("text", { x, y: 26, "text-anchor": "middle", "font-size": 11, "font-weight": 700, class: "muted" }, svg).textContent = t.toUpperCase();
    });
    const clip = (from, to) => { // point on target rect boundary along the centre line
      const dx = from.x - to.x, dy = from.y - to.y;
      const sx = (NW / 2 + 4) / Math.abs(dx || 1e-9), sy = (NH / 2 + 4) / Math.abs(dy || 1e-9);
      const s = Math.min(sx, sy);
      return { x: to.x + dx * s, y: to.y + dy * s };
    };
    const edgeEls = EDGES.map(([a, b, k]) => {
      const A = byId[a], B = byId[b];
      const p0 = clip(B, A), p1 = clip(A, B);
      const mx = (p0.x + p1.x) / 2;
      const d = Math.abs(A.x - B.x) < 5
        ? `M${p0.x},${p0.y} L${p1.x},${p1.y}`
        : `M${p0.x},${p0.y} C${mx},${p0.y} ${mx},${p1.y} ${p1.x},${p1.y}`;
      const el = MI.svg("path", { d, fill: "none", stroke: kindStyle[k].c, "stroke-width": 1.6, "stroke-dasharray": kindStyle[k].dash, "marker-end": `url(#cm-${k})`, class: "cm-edge" }, svg);
      return { a, b, k, el };
    });
    const nodeEls = {};
    NODES.forEach(n => {
      const g = MI.svg("g", { class: "cm-node", tabindex: 0, role: "button", "aria-label": n.label }, svg);
      MI.svg("rect", { x: n.x - NW / 2, y: n.y - NH / 2, width: NW, height: NH, rx: 9 }, g);
      MI.svg("text", { x: n.x, y: n.y + 4.5, "text-anchor": "middle", "font-size": 12.5 }, g).textContent = n.label;
      g.addEventListener("click", () => select(n.id));
      g.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(n.id); } });
      nodeEls[n.id] = g;
    });
    const verb = { b: "builds on", m: "is motivated by", t: "is tested by" };
    const verbOut = { b: "leads to", m: "motivates", t: "tests" };
    function select(id) {
      const n = byId[id];
      const nb = new Set([id]);
      edgeEls.forEach(e => { if (e.a === id || e.b === id) { nb.add(e.a); nb.add(e.b); } });
      Object.entries(nodeEls).forEach(([k, g]) => { g.classList.toggle("dim", !nb.has(k)); g.classList.toggle("sel", k === id); });
      edgeEls.forEach(e => e.el.classList.toggle("dim", !(e.a === id || e.b === id)));
      const ins = edgeEls.filter(e => e.b === id).map(e => `${verb[e.k]} <b>${byId[e.a].label}</b>`);
      const outs = edgeEls.filter(e => e.a === id).map(e => `${verbOut[e.k]} <b>${byId[e.b].label}</b>`);
      info.innerHTML = `<h4>${n.label}</h4><p>${n.d}</p>` +
        (ins.length ? `<p class="rel">It ${ins.join("; ")}.</p>` : "") +
        (outs.length ? `<p class="rel">It ${outs.join("; ")}.</p>` : "") +
        `<p class="go">Read more: ${chLink(n.ch)}</p>`;
    }
    document.getElementById("cmapReset").onclick = () => {
      Object.values(nodeEls).forEach(g => g.classList.remove("dim", "sel"));
      edgeEls.forEach(e => e.el.classList.remove("dim"));
      info.innerHTML = `<p class="hint">Click a box to see what it means and how it connects.</p>`;
    };
    select("superposition");
  }

  /* =========================================================
     2. Timeline (base MI_TIMELINE + extra entries)
     ========================================================= */
  const EXTRA = [
    { year: 1984, m: 6, tag: "theory", ch: 3, title: "Johnson–Lindenstrauss lemma", desc: "High-dimensional spaces hold exponentially many almost-orthogonal directions. The math behind superposition." },
    { year: 1996, m: 6, tag: "method", ch: 8, title: "Sparse coding of images", desc: "Olshausen & Field learn a sparse dictionary from image patches and get edge detectors like those in the visual cortex." },
    { year: 2006, m: 6, tag: "theory", ch: 3, title: "Compressed sensing", desc: "Sparse vectors can be recovered from few measurements. Superposition uses the same idea." },
    { year: 2013, m: 12, tag: "method", ch: 8, title: "k-sparse autoencoders", desc: "Makhzani & Frey keep only the k largest hidden units: the idea behind later TopK SAEs." },
    { year: 2018, m: 6, tag: "theory", ch: 8, title: "Word senses in superposition", desc: "Arora et al. show word embeddings are sums of sense vectors and recover them with sparse coding." },
    { year: 2021, m: 3, tag: "method", ch: 8, title: "Dictionary learning on BERT", desc: "Yun, Chen, Olshausen & LeCun find 'transformer factors' with sparse coding." },
    { year: 2022, m: 10, tag: "theory", ch: 3, title: "Polysemanticity and capacity", desc: "Scherlis et al. (Redwood) model how networks share neuron capacity between features." },
    { year: 2022, m: 11, tag: "beyond", ch: 10, title: "Chess knowledge in AlphaZero", desc: "McGrath et al. (PNAS) probe AlphaZero for human chess concepts across layers and training time." },
    { year: 2022, m: 12, tag: "method", ch: 8, title: "SAEs on toy superposition", desc: "Sharkey, Braun & beren show sparse autoencoders recover planted features (interim report)." },
    { year: 2023, m: 1, tag: "theory", ch: 3, title: "Superposition and double descent", desc: "Henighan et al. link memorization, superposition and double descent in toy models." },
    { year: 2023, m: 3, tag: "theory", ch: 10, title: "Quantization model of scaling", desc: "Michaud et al.: skills come in discrete 'quanta'; many quanta give smooth power-law scaling." },
    { year: 2023, m: 5, tag: "lm", ch: 3, title: "Finding neurons in a haystack", desc: "Gurnee et al. use sparse probing to find single neurons and small neuron groups for features." },
    { year: 2023, m: 5, tag: "beyond", ch: 10, title: "Brain-inspired modular training", desc: "Liu, Gan & Tegmark penalize long connections so networks become visibly modular." },
    { year: 2023, m: 8, tag: "theory", ch: 10, title: "Local learning coefficient", desc: "Lau et al. measure the effective complexity of a trained network using singular learning theory." },
    { year: 2023, m: 8, tag: "lm", ch: 10, title: "Linear relation decoding", desc: "Hernandez et al.: many relations ('plays instrument') act like one affine map on the subject state." },
    { year: 2023, m: 9, tag: "lm", ch: 10, title: "Attention sinks", desc: "Xiao et al. find heads dump attention on the first token; keeping it lets models stream long text." },
    { year: 2023, m: 10, tag: "beyond", ch: 10, title: "Maze agent & cheese vector", desc: "Mini, Turner et al. steer a maze-solving RL policy by subtracting a 'cheese' activation vector." },
    { year: 2023, m: 10, tag: "vision", ch: 10, title: "Decomposing CLIP with text", desc: "Gandelsman, Efros & Steinhardt label CLIP heads with text: colour heads, location heads, counting heads." },
    { year: 2024, m: 1, tag: "method", ch: 10, title: "Patchscopes", desc: "Ghandeharioun et al. patch a hidden state into a prompt so the model describes it in words." },
    { year: 2024, m: 2, tag: "theory", ch: 10, title: "Stagewise development", desc: "Hoogland et al. use the LLC to find developmental stages in small transformers." },
    { year: 2024, m: 2, tag: "lm", ch: 10, title: "Massive activations", desc: "Sun et al.: a few activations are huge, nearly constant, and act like bias terms." },
    { year: 2024, m: 2, tag: "beyond", ch: 10, title: "MIPS: program synthesis", desc: "Michaud et al. turn small trained RNNs into Python code via mechanistic interpretability." },
    { year: 2024, m: 4, tag: "beyond", ch: 10, title: "Kolmogorov–Arnold Networks", desc: "Liu et al. put learnable 1-D functions on edges, so small models can be read as formulas." },
    { year: 2024, m: 5, tag: "theory", ch: 3, title: "Not all features are linear", desc: "Engels et al. find circular features for days of the week and months." },
    { year: 2024, m: 6, tag: "beyond", ch: 10, title: "Learned look-ahead in chess", desc: "Jenner et al. show Leela Chess Zero's policy network represents future moves." },
    { year: 2024, m: 6, tag: "lm", ch: 10, title: "Confidence regulation neurons", desc: "Stolfo et al. find entropy neurons and token-frequency neurons that tune output confidence." },
    { year: 2024, m: 6, tag: "safety", ch: 10, title: "TalkTuner", desc: "Chen et al. show users a chatbot's internal model of their age, gender and status, and let them edit it." },
    { year: 2024, m: 9, tag: "method", ch: 8, title: "Feature absorption", desc: "Chanin et al. show SAE features can silently fail when a more specific feature absorbs their job." },
    { year: 2024, m: 10, tag: "method", ch: 10, title: "Bilinear MLPs", desc: "Pearce et al. use MLPs without element-wise nonlinearity, so features can be read from weights." },
    { year: 2024, m: 10, tag: "theory", ch: 3, title: "The geometry of concepts", desc: "Li, Michaud, Tegmark et al. study crystal-like and lobe-like structure in SAE features." },
    { year: 2024, m: 12, tag: "beyond", ch: 10, title: "InterPLM", desc: "Simon & Zou train SAEs on the protein model ESM-2 and match features to biological concepts." },
    { year: 2025, m: 1, tag: "method", ch: 8, title: "SAE seed instability", desc: "Paulo & Belrose: SAEs trained with different seeds share only about 30% of features." },
    { year: 2025, m: 1, tag: "method", ch: 8, title: "SAEs on random transformers", desc: "Heap et al.: common auto-interpretability scores barely tell SAEs on trained and random transformers apart." },
    { year: 2025, m: 2, tag: "beyond", ch: 10, title: "Interpreting Evo 2", desc: "Goodfire and Arc Institute find features for exons, protein structure and viral DNA in a DNA model." },
    { year: 2025, m: 3, tag: "beyond", ch: 10, title: "Teaching grandmasters", desc: "Schut et al. extract new AlphaZero concepts and teach them to four chess grandmasters." },
    { year: 2025, m: 3, tag: "method", ch: 8, title: "SAEBench", desc: "A shared benchmark that scores SAEs on many practical metrics, not only reconstruction." },
    { year: 2025, m: 6, tag: "lm", ch: 10, title: "Thought anchors", desc: "Bogdan, Macar, Nanda & Conmy find a few planning and backtracking sentences steer long reasoning." },
    { year: 2026, m: 1, tag: "beyond", ch: 10, title: "Interpretability finds a biomarker", desc: "Goodfire and Prima Mente find an Alzheimer's-detection model relies on DNA fragment length, pointing to a new class of biomarker." },
  ];
  // Shared entries carry a month (m). Spread any that lack one across the year.
  const base = (window.MI_TIMELINE || []).map((d, i, arr) => {
    if (d.m) return { ...d, base: true };
    const same = arr.filter(e => e.year === d.year && !e.m);
    return { ...d, m: 1 + (11 * (same.indexOf(d) + 0.5)) / same.length, base: true };
  });
  // Skip any page-only entry whose title is already in the shared timeline.
  const baseTitles = new Set(base.map(d => d.title.toLowerCase()));
  const ALL = [...base, ...EXTRA.filter(d => !baseTitles.has(d.title.toLowerCase()))]
    .map((d, i) => ({ ...d, i }))
    .sort((a, b) => a.year - b.year || a.m - b.m || a.i - b.i);
  ALL.forEach((d, i) => (d.id = "tl-" + i));

  const tagColor = { vision: "var(--green)", lm: "var(--blue)", method: "var(--purple)", theory: "var(--yellow)", safety: "var(--red)", beyond: "var(--accent)" };
  const tagName = { vision: "vision", lm: "language models", method: "methods", theory: "theory", safety: "safety & use", beyond: "beyond text" };

  // era band
  const band = document.getElementById("eraBand");
  const bandTip = document.getElementById("eraTip");
  const Y0 = 2009, Y1 = 2026, W = 960, L = 60, R = 20;
  const xOf = y => L + ((y - Y0) / (Y1 - Y0)) * (W - L - R);
  function drawBand(filterFn) {
    band.innerHTML = "";
    const eras = [[2009, 2012.5, "Roots"], [2012.5, 2019.5, "Inside vision models"], [2019.5, 2021.5, "Circuits"], [2021.5, 2023.5, "LM circuits"], [2023.5, 2026.5, "Scale & use"]];
    eras.forEach(([a, b, t], i) => {
      const x0 = i === 0 ? 0 : xOf(a), x1 = Math.min(W, xOf(b));
      MI.svg("rect", { x: x0, y: 0, width: x1 - x0, height: 196, fill: i % 2 ? "var(--bg-2)" : "var(--panel)" }, band);
      MI.svg("text", { x: (x0 + x1) / 2, y: 16, "text-anchor": "middle", "font-size": 11, "font-weight": 700, class: "muted" }, band).textContent = t.toUpperCase();
    });
    MI.svg("line", { x1: 0, y1: 168, x2: W, y2: 168, stroke: "var(--rule)" }, band);
    for (let y = Y0; y <= Y1; y++) {
      MI.svg("line", { x1: xOf(y), y1: 165, x2: xOf(y), y2: 171, stroke: "var(--ink-3)" }, band);
      if (y % 2 === 1 || y === Y1) MI.svg("text", { x: xOf(y), y: 186, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, band).textContent = y;
    }
    MI.svg("text", { x: 22, y: 186, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, band).textContent = "earlier";
    const counts = {};
    ALL.forEach(d => {
      const yr = d.year < Y0 ? null : d.year;
      const cx = yr === null ? 22 : xOf(d.year);
      const key = yr === null ? "early" : d.year;
      counts[key] = (counts[key] || 0) + 1;
      const stackX = (counts[key] - 1) % 2, stackY = Math.floor((counts[key] - 1) / 2);
      const on = filterFn(d);
      const c = MI.svg("circle", { cx: cx + (stackX ? 6 : 0) - 3, cy: 156 - stackY * 9, r: 3.6, fill: on ? tagColor[d.tag] : "var(--rule)", class: "tl-dot" }, band);
      c.addEventListener("mouseenter", () => { bandTip.innerHTML = `<b>${d.year} · ${d.title}</b> — ${d.desc}`; });
      c.addEventListener("click", () => { const t = document.getElementById(d.id); if (t) { t.scrollIntoView({ block: "center" }); t.classList.add("flash"); setTimeout(() => t.classList.remove("flash"), 1200); } });
    });
  }

  // filters + list
  const tagBox = document.getElementById("tlTags"), chSel = document.getElementById("tlCh"), q = document.getElementById("tlQ"), list = document.getElementById("tlList"), count = document.getElementById("tlCount");
  let tag = "all";
  ["all", ...Object.keys(tagName)].forEach(t => {
    const b = MI.el("button", { text: t === "all" ? "all topics" : tagName[t] }, tagBox);
    b.dataset.t = t;
    if (t !== "all") b.style.setProperty("--dot", tagColor[t]);
    b.onclick = () => { tag = t; render(); };
  });
  (window.MI_CHAPTERS || []).filter(c => c.n > 0 && c.n < 12).forEach(c => MI.el("option", { value: c.n, text: `Ch. ${c.n} · ${c.title}` }, chSel));
  chSel.onchange = render; q.oninput = render;
  function render() {
    [...tagBox.children].forEach(b => b.classList.toggle("on", b.dataset.t === tag));
    const ch = +chSel.value, s = q.value.trim().toLowerCase();
    const f = d => (tag === "all" || d.tag === tag) && (!ch || d.ch === ch) && (!s || (d.title + " " + d.desc).toLowerCase().includes(s));
    drawBand(f);
    list.innerHTML = "";
    let last = null, n = 0;
    ALL.filter(f).forEach(d => {
      n++;
      if (d.year !== last) { MI.el("div", { class: "tl-year", text: d.year }, list); last = d.year; }
      const c = CH(d.ch);
      MI.el("div", { class: "tl-item", id: d.id, html: `<span class="tl-tag" style="--dot:${tagColor[d.tag]}">${tagName[d.tag]}</span><b>${d.title}</b> <span class="d">— ${d.desc}</span>${c ? ` <a class="ch" href="${c.file}">Ch. ${c.n} →</a>` : ""}` }, list);
    });
    count.textContent = `${n} of ${ALL.length} entries`;
  }
  render();
  MI.onTheme(render);

  /* =========================================================
     3. Glossary
     ========================================================= */
  const G = {};
  /* All terms live in the shared glossary (assets/glossary.js). Show each entry once:
     skip aliases (same object), plain plurals whose singular exists, and "sae". */
  const src = window.MI_GLOSSARY || {};
  const seen = new Set();
  Object.entries(src).forEach(([k, v]) => {
    if (seen.has(v)) return;
    if (k.endsWith("s") && src[k.slice(0, -1)]) return;
    if (k === "sae") return; // covered by "Sparse autoencoder (SAE)"
    seen.add(v);
    const key = v.name.toLowerCase();
    if (!G[key]) G[key] = v;
  });
  const entries = Object.values(G).sort((a, b) => a.name.localeCompare(b.name, "en", { sensitivity: "base" }));
  const gq = document.getElementById("gQ"), gl = document.getElementById("gList"), ga = document.getElementById("gAlpha"), gc = document.getElementById("gCount");
  const letters = [...new Set(entries.map(e => e.name[0].toUpperCase()))];
  letters.forEach(L => {
    const a = MI.el("a", { href: "#g-" + L, text: L }, ga);
    a.onclick = e => { e.preventDefault(); const t = document.getElementById("g-" + L); if (t) gl.scrollTop = t.offsetTop - gl.offsetTop; };
  });
  function renderG() {
    const s = gq.value.trim().toLowerCase();
    gl.innerHTML = "";
    let last = null, n = 0;
    entries.filter(e => !s || e.name.toLowerCase().includes(s) || e.def.toLowerCase().includes(s)).forEach(e => {
      n++;
      const L = e.name[0].toUpperCase();
      if (L !== last) { MI.el("div", { class: "g-letter", id: "g-" + L, text: L }, gl); last = L; }
      const hl = t => s ? t.replace(new RegExp("(" + s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>") : t;
      MI.el("div", { class: "g-item", html: `<dt>${hl(e.name)}</dt><dd>${hl(e.def)}</dd>` }, gl);
    });
    gc.textContent = `${n} of ${entries.length} terms`;
    if (!n) MI.el("p", { class: "muted-p", text: "No term matches. Try a shorter word." }, gl);
  }
  gq.oninput = renderG;
  renderG();
})();
