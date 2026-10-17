/* Chapter 11 widgets: theories of impact map, people directory, debate explorer,
   dark-matter schematic, persona dots, open-problems map and checklist. */
(function () {
  const { svg: S, el: E } = window.MI;

  /* ---------------- helper: clickable box map ---------------- */
  function boxMap(svgEl, detailEl, groups, render) {
    svgEl.innerHTML = "";
    const nodes = [];
    groups.forEach(g => {
      S("text", { x: g.x, y: 22, "font-size": 12.5, "font-weight": 800, fill: g.color }, svgEl).textContent = g.title;
      g.items.forEach((it, i) => {
        const y = 34 + i * (g.h || 58);
        const node = S("g", { class: "node", tabindex: 0, role: "button" }, svgEl);
        S("rect", { x: g.x, y, width: g.w, height: (g.h || 58) - 10, rx: 9, fill: g.soft, stroke: g.color, "stroke-width": 1.4 }, node);
        const lines = it.label.split("\n");
        lines.forEach((ln, k) => {
          S("text", { x: g.x + g.w / 2, y: y + ((g.h || 58) - 10) / 2 + 5 + (k - (lines.length - 1) / 2) * 16, "text-anchor": "middle", "font-size": 13 }, node).textContent = ln;
        });
        const pick = () => {
          nodes.forEach(n => n.classList.remove("sel"));
          node.classList.add("sel");
          detailEl.innerHTML = render(it, g);
        };
        node.addEventListener("click", pick);
        node.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
        nodes.push(node);
      });
    });
  }
  const badge = lvl => lvl === "strong" ? '<span class="badge strong">real evidence</span>'
    : lvl === "some" ? '<span class="badge some">early evidence</span>' : '<span class="badge none">mostly a hope</span>';

  /* ---------------- 1. Theories of impact ---------------- */
  const toiGroups = [
    { title: "DETECT", x: 10, w: 155, color: "var(--red)", soft: "var(--red-soft)", items: [
      { label: "Audit before\nrelease", lvl: "strong", def: "Look inside a model before release to find hidden goals, backdoors or dangerous knowledge.", ev: "Blind auditing game: 3 of 4 teams found a planted hidden objective (2025). Every major Anthropic system card since Sept 2025 has a white-box audit; in 2026 these used attribution graphs and an NLA investigator agent. Caveat: in AuditBench (2026), white-box tools helped an AI auditor less than black-box tools." },
      { label: "Monitor in\ndeployment", lvl: "strong", def: "Run cheap checks on activations while the model is in use, for misuse or bad behaviour.", ev: "Linear probes detect harmful intent and backdoored 'sleeper agent' behaviour very well. In Jan 2026, Google DeepMind (Gemini) and Anthropic (Constitutional Classifiers++) described probes in production (Chapter 7). Caveat: 'Neural Chameleons' (Dec 2025) showed that models can be trained to evade unseen probes." },
      { label: "Detect\ndeception", lvl: "some", def: "Tell when a model knowingly says false things or hides its goals.", ev: "Deception probes work in test settings (Chapter 7). Nanda (2025) warns that interpretability will not reliably find deceptive AI on its own." },
      { label: "Evidence for\ndecision makers", lvl: "some", def: "Give labs and governments concrete evidence about risks, not only behaviour scores.", ev: "System-card findings on evaluation awareness changed how labs read their test results. In 2026, white-box tools found test awareness that models did not say out loud (Mythos Preview, Opus 5 cards). Wider policy use is still limited." },
    ]},
    { title: "IMPROVE", x: 180, w: 155, color: "var(--blue)", soft: "var(--blue-soft)", items: [
      { label: "Debug and fix\nbehaviour", lvl: "some", def: "Find the mechanism behind a failure and fix the cause.", ev: "Persona features explained emergent misalignment and steering suppressed it (2025). Transluce's Monitor reduced the '9.8 < 9.11' error by suppressing neuron groups tied to Bible verses and dates. An attribution graph explained a misreported tool result in the Opus 4.6 audit (2026)." },
      { label: "Steer and\ncontrol", lvl: "strong", def: "Change behaviour at run time by adding or removing directions.", ev: "Refusal direction, persona vectors, Golden Gate Claude, eval-awareness steering (Chapters 7–8)." },
      { label: "Better\ntraining", lvl: "some", def: "Use interpretability during training, for example to prevent unwanted traits.", ev: "'Preventative steering' with persona vectors during fine-tuning (2025). Goodfire used hallucination probes as RL rewards (2026). Training directly against interpretability signals is risky: the model may learn to hide them." },
    ]},
    { title: "PREDICT", x: 350, w: 155, color: "var(--purple)", soft: "var(--purple-soft)", items: [
      { label: "Watch\ntraining", lvl: "some", def: "Notice when important structure forms during training, for example a phase change.", ev: "Induction heads form in a visible phase change (Chapter 4); developmental interpretability tracks stages with the learning coefficient (Chapter 10)." },
      { label: "Enumerative\nsafety", lvl: "none", def: "List every feature and check that none is dangerous.", ev: "Seed dependence, feature splitting and 'dark matter' make a complete list look out of reach for now." },
      { label: "Predict new\nbehaviour", lvl: "some", def: "Predict how a model will act in new situations from its internals.", ev: "The toxic-persona latent predicted emergent misalignment before it showed in behaviour. Broader prediction is still rare." },
    ]},
    { title: "SCIENCE", x: 520, w: 150, color: "var(--green)", soft: "var(--green-soft)", items: [
      { label: "Understand\nlearning", lvl: "strong", def: "Learn how neural networks represent and compute: superposition, circuits, world models.", ev: "Superposition toy models, induction heads, grokking, Othello-GPT, attribution graphs." },
      { label: "Microscope AI", lvl: "some", def: "Learn new facts about the world from what a model has learned.", ev: "SAE features on DNA and protein models led to biological findings (Chapter 10). AlphaZero concepts were taught to chess grandmasters." },
    ]},
  ];
  const toiSvg = document.getElementById("toiSvg");
  if (toiSvg) boxMap(toiSvg, document.getElementById("toiDetail"), toiGroups,
    (it, g) => `<h4>${it.label.replace("\n", " ")} ${badge(it.lvl)}</h4>${it.def}<div class="ev"><b>Evidence:</b> ${it.ev}</div>`);

  /* ---------------- 2. People directory ---------------- */
  const T = { vision: "vision circuits", transformer: "transformer circuits", sae: "SAEs & dictionaries", causal: "causal methods", probing: "probing & steering", theory: "theory & agendas", safety: "safety applications", tools: "tools & training" };
  const chipClass = { vision: "vision", transformer: "lm", sae: "method", causal: "method", probing: "lm", theory: "theory", safety: "safety", tools: "" };
  const L = (t, u) => u ? `<a href="${u}">${t}</a>` : t;
  const P = [
    { n: "Chris Olah", a: "Anthropic co-founder, interpretability lead; earlier Google Brain, OpenAI, Distill", t: ["vision", "transformer", "sae", "theory"], c: [
      L("Feature Visualization (2017)", "https://distill.pub/2017/feature-visualization/"), L("Zoom In: An Introduction to Circuits (2020)", "https://distill.pub/2020/circuits/zoom-in/"),
      L("Variables & interpretable bases essay (2022)", "https://transformer-circuits.pub/2022/mech-interp-essay/index.html"), L("Interpretability Dreams (2023)", "https://transformer-circuits.pub/2023/interpretability-dreams/index.html"),
      L("A Toy Model of Mechanistic (Un)Faithfulness (2025)", "https://transformer-circuits.pub/2025/faithfulness-toy-model/index.html"), L("The Persona Selection Model (2026, with Marks & Lindsey)", "https://alignment.anthropic.com/2026/psm")] },
    { n: "Nelson Elhage", a: "Anthropic (lead author of key early papers)", t: ["transformer", "theory"], c: [
      L("A Mathematical Framework for Transformer Circuits (2021)", "https://transformer-circuits.pub/2021/framework/index.html"), L("Toy Models of Superposition (2022)", "https://transformer-circuits.pub/2022/toy_model/index.html"),
      "Softmax Linear Units (2022)", L("Privileged Bases in the Transformer Residual Stream (2023)", "https://transformer-circuits.pub/2023/privileged-basis/index.html")] },
    { n: "Catherine Olsson", a: "Anthropic (as of these papers)", t: ["transformer"], c: [L("In-context Learning and Induction Heads (2022)", "https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html")] },
    { n: "Trenton Bricken", a: "Anthropic", t: ["sae", "safety"], c: [
      L("Towards Monosemanticity (2023)", "https://transformer-circuits.pub/2023/monosemantic-features/index.html"), "Attention Approximates Sparse Distributed Memory (2021)",
      L("Auditing Language Models for Hidden Objectives (2025, co-author)", "https://arxiv.org/abs/2503.10965")] },
    { n: "Adly Templeton", a: "Anthropic", t: ["sae"], c: [L("Scaling Monosemanticity (2024)", "https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html")] },
    { n: "Josh Batson", a: "Anthropic", t: ["sae", "transformer", "theory"], c: [
      L("Circuit Tracing & Biology of an LLM (2025, co-lead)", "https://transformer-circuits.pub/2025/attribution-graphs/biology.html"), L("Sparse Crosscoders (2024)", "https://transformer-circuits.pub/2024/crosscoders/index.html"),
      L("Open Problems in Mech Interp (2025)", "https://arxiv.org/abs/2501.16496"), L("Characterizing interference weights in a tiny LM (2026, with Turner & Wu)", "https://transformer-circuits.pub/2026/interference_effectiveness_helpfulness/index.html")] },
    { n: "Emmanuel Ameisen", a: "Anthropic", t: ["transformer", "tools"], c: [
      L("Circuit Tracing: Revealing Computational Graphs (2025, lead)", "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"), L("Mentored the open-source circuit-tracer (2025)", "https://github.com/safety-research/circuit-tracer")] },
    { n: "Jack Lindsey", a: "Anthropic", t: ["sae", "transformer", "safety", "probing"], c: [
      L("Sparse Crosscoders & model diffing (2024)", "https://transformer-circuits.pub/2024/crosscoders/index.html"), L("On the Biology of a Large Language Model (2025)", "https://transformer-circuits.pub/2025/attribution-graphs/biology.html"),
      L("Emergent Introspective Awareness (2025)", "https://transformer-circuits.pub/2025/introspection/index.html"), L("Persona Vectors (2025, co-author)", "https://arxiv.org/abs/2507.21509"),
      L("Emotion Concepts and their Function (2026, senior author)", "https://transformer-circuits.pub/2026/emotions/index.html"), L("The Assistant Axis (2026, co-author)", "https://arxiv.org/abs/2601.10387")] },
    { n: "Samuel Marks", a: "Anthropic; earlier Bau lab and Tegmark group", t: ["probing", "sae", "safety"], c: [
      L("The Geometry of Truth (2023)", "https://arxiv.org/abs/2310.06824"), L("Sparse Feature Circuits (2024)", "https://arxiv.org/abs/2403.19647"),
      L("Auditing Language Models for Hidden Objectives (2025)", "https://arxiv.org/abs/2503.10965"), L("Auditing agents (2025, senior author)", "https://alignment.anthropic.com/2025/automated-auditing/"),
      L("AuditBench (2026, co-author)", "https://arxiv.org/abs/2602.22755"), L("The Persona Selection Model (2026)", "https://alignment.anthropic.com/2026/psm")] },
    { n: "Wes Gurnee", a: "Anthropic; earlier MIT (Tegmark group)", t: ["probing", "transformer", "theory"], c: [
      L("Finding Neurons in a Haystack (2023)", "https://arxiv.org/abs/2305.01610"), L("Language Models Represent Space and Time (2023)", "https://arxiv.org/abs/2310.02207"),
      L("Universal Neurons in GPT-2 (2024)", "https://arxiv.org/abs/2401.12181"), L("Verbalizable representations form a global workspace (2026) — see Ch. 10", "https://transformer-circuits.pub/2026/workspace/index.html")] },
    { n: "Neel Nanda", a: "Leads Google DeepMind's mech interp team; earlier Anthropic, independent", t: ["transformer", "sae", "causal", "theory", "tools", "safety"], c: [
      L("Progress Measures for Grokking (2023)", "https://arxiv.org/abs/2301.05217"), L("TransformerLens (2022)", "https://github.com/TransformerLensOrg/TransformerLens"),
      L("A Longlist of Theories of Impact (2022)", "https://www.alignmentforum.org/posts/uK6sQCNMw8WKzJeCQ/a-longlist-of-theories-of-impact-for-interpretability"), "200 Concrete Open Problems (2022–23)",
      L("Gemma Scope (2024)", "https://arxiv.org/abs/2408.05147"), L("Interpretability Will Not Reliably Find Deceptive AI (2025)", "https://www.lesswrong.com/posts/PwnadG4BFjaER3MGf/interpretability-will-not-reliably-find-deceptive-ai"),
      L("A Pragmatic Vision for Interpretability (Dec 2025, lead)", "https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability")] },
    { n: "Tom McGrath", a: "Goodfire co-founder, chief scientist; earlier Google DeepMind", t: ["causal", "theory"], c: [
      L("Acquisition of Chess Knowledge in AlphaZero (2022)", "https://arxiv.org/abs/2111.09259"), L("The Hydra Effect: self-repair (2023)", "https://arxiv.org/abs/2307.15771"), "Interpreting Evo 2 (2025, co-author) — see Ch. 10"] },
    { n: "Arthur Conmy", a: "Google DeepMind", t: ["causal", "transformer", "sae"], c: [
      L("ACDC: Automated Circuit DisCovery (2023)", "https://arxiv.org/abs/2304.14997"), L("Copy Suppression (2023, co-author)", "https://arxiv.org/abs/2310.04625"), L("Gemma Scope (2024, co-author)", "https://arxiv.org/abs/2408.05147")] },
    { n: "Senthooran Rajamanoharan", a: "Google DeepMind", t: ["sae", "safety"], c: [
      L("Gated SAEs (2024)", "https://arxiv.org/abs/2404.16014"), L("JumpReLU SAEs (2024)", "https://arxiv.org/abs/2407.14435"), L("Convergent linear representations of emergent misalignment (2025, co-author)", "https://arxiv.org/abs/2506.11618")] },
    { n: "János Kramár", a: "Google DeepMind", t: ["causal", "sae"], c: [L("AtP*: efficient attribution patching (2024)", "https://arxiv.org/abs/2403.00745"), "Gated & JumpReLU SAEs, Gemma Scope (co-author)", L("Building Production-Ready Probes for Gemini (2026, lead)", "https://arxiv.org/abs/2601.11516")] },
    { n: "Callum McDougall", a: "Founder of ARENA; Google DeepMind", t: ["tools", "transformer"], c: ["ARENA interpretability curriculum", L("Copy Suppression (2023)", "https://arxiv.org/abs/2310.04625"), "sae_vis feature dashboards", L("Gemma Scope 2 (Dec 2025, lead)", "https://www.lesswrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2")] },
    { n: "Been Kim", a: "Google DeepMind", t: ["vision", "theory", "probing"], c: [
      L("TCAV: concept activation vectors (2018)", "https://arxiv.org/abs/1711.11279"), L("Sanity Checks for Saliency Maps (2018, co-author)", "https://arxiv.org/abs/1810.03292"),
      "AlphaZero concepts & teaching grandmasters (2022, 2025)", L("Neologisms position paper (2025)", "https://arxiv.org/abs/2502.07586")] },
    { n: "Asma Ghandeharioun", a: "Google", t: ["probing"], c: [L("Patchscopes (2024)", "https://arxiv.org/abs/2401.06102"), "— see self-explanation in Ch. 10"] },
    { n: "Leo Gao", a: "OpenAI (as of these papers)", t: ["sae"], c: [L("Scaling and Evaluating Sparse Autoencoders (TopK, 2024)", "https://arxiv.org/abs/2406.04093"), L("Weight-sparse transformers have interpretable circuits (2025)", "https://arxiv.org/abs/2511.13653")] },
    { n: "Dan Mossing", a: "OpenAI interpretability (2025 papers); listed with Anthropic on the 2026 NLA paper", t: ["sae", "safety"], c: [L("Persona Features Control Emergent Misalignment (2025, senior author)", "https://arxiv.org/abs/2506.19823"), L("Weight-sparse transformers (2025, co-author)", "https://arxiv.org/abs/2511.13653"), L("Natural Language Autoencoders (2026, co-author)", "https://transformer-circuits.pub/2026/nla/index.html")] },
    { n: "Max Tegmark, Eric Michaud, Ziming Liu", a: "MIT, Tegmark group (as of these papers)", t: ["theory", "probing"], c: [
      L("The Quantization Model of Neural Scaling (2023)", "https://arxiv.org/abs/2303.13506"), L("KAN: Kolmogorov–Arnold Networks (2024)", "https://arxiv.org/abs/2404.19756"),
      L("Brain-inspired modular training (2023)", "https://arxiv.org/abs/2305.08746"), L("Geometry of Truth (2023, Tegmark)", "https://arxiv.org/abs/2310.06824")] },
    { n: "David Bau", a: "Northeastern University", t: ["vision", "causal", "tools"], c: [
      L("Network Dissection (2017)", "https://arxiv.org/abs/1704.05796"), L("ROME: locating and editing facts (2022)", "https://arxiv.org/abs/2202.05262"),
      L("Function Vectors (2023)", "https://arxiv.org/abs/2310.15213"), L("NNsight & NDIF (2024)", "https://arxiv.org/abs/2407.14561")] },
    { n: "Yonatan Belinkov", a: "Technion", t: ["probing", "causal"], c: [
      "Probing Classifiers: Promises, Shortcomings, and Advances (survey)", L("Causal mediation analysis of gender bias (2020, co-author)", "https://arxiv.org/abs/2004.12265"),
      L("Sparse Feature Circuits (2024, co-author)", "https://arxiv.org/abs/2403.19647"), "MIB benchmark (2025)"] },
    { n: "Mor Geva", a: "Tel Aviv University", t: ["transformer", "causal"], c: [
      L("FFN layers are key-value memories (2021)", "https://arxiv.org/abs/2012.14913"), L("Dissecting Recall of Factual Associations (2023)", "https://arxiv.org/abs/2304.14767"), L("Patchscopes (2024, co-author)", "https://arxiv.org/abs/2401.06102")] },
    { n: "Atticus Geiger & Christopher Potts", a: "Potts: Stanford. Geiger: Pr(Ai)²R group (earlier Stanford)", t: ["causal", "theory", "tools"], c: [
      L("Causal Abstractions of Neural Networks (2021)", "https://arxiv.org/abs/2106.02997"), L("Distributed Alignment Search (2023)", "https://arxiv.org/abs/2303.02536"),
      L("pyvene (2024)", "https://arxiv.org/abs/2403.07809"), L("Causal Abstraction: A Theoretical Foundation (JMLR 2025)", "https://www.jmlr.org/papers/v26/23-0058.html")] },
    { n: "Martin Wattenberg & Fernanda Viégas", a: "Harvard; Google (PAIR)", t: ["theory", "probing"], c: [
      L("How to Use t-SNE Effectively (2016)", "https://distill.pub/2016/misread-tsne/"), L("Visualizing and Measuring the Geometry of BERT (2019)", "https://arxiv.org/abs/1906.02715"),
      "Othello-GPT, ITI (co-authors)", L("TalkTuner dashboard (2024)", "https://arxiv.org/abs/2406.07882")] },
    { n: "Jacob Andreas", a: "MIT", t: ["vision", "theory"], c: [
      L("Compositional Explanations of Neurons (2020)", "https://arxiv.org/abs/2006.14032"), L("MILAN: natural-language neuron descriptions (2022)", "https://arxiv.org/abs/2201.11114"), "Language Models as Agent Models (2022)"] },
    { n: "Jacob Steinhardt", a: "UC Berkeley; Transluce co-founder", t: ["transformer", "probing", "safety"], c: [
      L("IOI circuit (2022, co-author)", "https://arxiv.org/abs/2211.00593"), L("Discovering Latent Knowledge / CCS (2022, advisor)", "https://arxiv.org/abs/2212.03827"),
      L("Predictive Concept Decoders (2025)", "https://arxiv.org/abs/2512.15712"), L("Scaling Activation Oracles to Trillion-Parameter Models (2026)", "https://transluce.org/scaling-activation-oracles"), "Transluce: Monitor, investigator agents, Docent"] },
    { n: "Sarah Schwettmann", a: "Transluce co-founder", t: ["vision", "safety", "tools"], c: [L("MILAN (2022)", "https://arxiv.org/abs/2201.11114"), L("MAIA: an automated interpretability agent (2024, MIT)", "https://arxiv.org/abs/2404.14394"), L("Transluce Monitor (2024)", "https://transluce.org/observability-interface"), L("Language model circuits are sparse in the neuron basis (2025)", "https://transluce.org/neuron-circuits")] },
    { n: "Nora Belrose", a: "EleutherAI", t: ["probing", "sae", "theory"], c: [
      L("Tuned Lens (2023)", "https://arxiv.org/abs/2303.08112"), L("LEACE concept erasure (2023)", "https://arxiv.org/abs/2306.03819"),
      L("SAEs on the same data learn different features (2025)", "https://arxiv.org/abs/2501.16615"), L("Networks learn statistics of increasing complexity (2024)", "https://arxiv.org/abs/2402.04362")] },
    { n: "Lee Sharkey", a: "Goodfire; earlier Apollo co-founder", t: ["sae", "theory"], c: [
      "Early SAE interim report (2022)", L("Open Problems in Mech Interp (2025, lead)", "https://arxiv.org/abs/2501.16496"), L("Attribution-based Parameter Decomposition (2025, Apollo)", "https://arxiv.org/abs/2501.14926"),
      L("Stochastic Parameter Decomposition (2025, Goodfire)", "https://arxiv.org/abs/2506.20790"), "Bilinear MLPs (2025)"] },
    { n: "Lucius Bushnaq & Dan Braun", a: "Apollo Research (e2e SAEs, APD); then Goodfire (SPD)", t: ["sae", "theory"], c: [
      L("End-to-end sparse dictionary learning (2024)", "https://arxiv.org/abs/2405.12241"), L("APD (Jan 2025, Apollo)", "https://arxiv.org/abs/2501.14926"), L("Stochastic Parameter Decomposition (June 2025, Goodfire)", "https://arxiv.org/abs/2506.20790")] },
    { n: "Marius Hobbhahn", a: "Apollo Research CEO", t: ["safety", "probing"], c: [L("Detecting Strategic Deception Using Linear Probes (2025, co-author)", "https://arxiv.org/abs/2502.03407"), "Evaluations for scheming and deception"] },
    { n: "Stefan Heimersheim", a: "Apollo Research (as of these papers)", t: ["causal", "transformer"], c: [L("How to use and interpret activation patching (2024)", "https://arxiv.org/abs/2404.15255"), "A circuit for Python docstrings (2023)", "APD (co-author)"] },
    { n: "Joseph Bloom", a: "Decode Research (SAELens); later UK AI Security Institute", t: ["sae", "tools"], c: [L("SAELens library", "https://github.com/jbloomAus/SAELens"), "Open GPT-2 small residual SAEs (2024)", "Feature absorption (co-author)"] },
    { n: "Johnny Lin", a: "Neuronpedia", t: ["tools"], c: [L("Neuronpedia", "https://www.neuronpedia.org"), "Attribution-graph frontend for circuit-tracer (2025)"] },
    { n: "Hoagy Cunningham & Logan Riggs", a: "", t: ["sae", "safety"], c: [L("Sparse Autoencoders Find Highly Interpretable Features (2023)", "https://arxiv.org/abs/2309.08600"), L("Decomposing the Dark Matter of SAEs (2024, Riggs)", "https://arxiv.org/abs/2410.14670"),
      L("Constitutional Classifiers++: probes in production (2026, Cunningham lead, Anthropic)", "https://arxiv.org/abs/2601.04603")] },
    { n: "Kevin Wang", a: "", t: ["transformer", "causal"], c: [L("Interpretability in the Wild: the IOI circuit (2022, lead)", "https://arxiv.org/abs/2211.00593")] },
    { n: "Michael Hanna", a: "", t: ["transformer", "tools"], c: [L("How does GPT-2 compute greater-than? (2023)", "https://arxiv.org/abs/2305.00586"), L("circuit-tracer (2025)", "https://github.com/safety-research/circuit-tracer")] },
    { n: "Kenneth Li", a: "", t: ["probing", "transformer"], c: [L("Emergent World Representations: Othello-GPT (2022)", "https://arxiv.org/abs/2210.13382"), "Inference-Time Intervention (2023)"] },
    { n: "Kevin Meng", a: "", t: ["causal"], c: [L("ROME: Locating and Editing Factual Associations (2022)", "https://arxiv.org/abs/2202.05262"), "MEMIT (2022)"] },
    { n: "Andy Arditi", a: "", t: ["probing", "safety"], c: [L("Refusal is mediated by a single direction (2024)", "https://arxiv.org/abs/2406.11717"), L("Persona Vectors (2025, co-author)", "https://arxiv.org/abs/2507.21509")] },
    { n: "Collin Burns", a: "", t: ["probing"], c: [L("Discovering Latent Knowledge without Supervision (CCS, 2022)", "https://arxiv.org/abs/2212.03827")] },
    { n: "Alex Turner", a: "", t: ["probing", "safety"], c: [L("Activation Addition / steering vectors (2023)", "https://arxiv.org/abs/2308.10248"), "Maze-solving policy 'cheese vector' (2023) — see Ch. 10"] },
    { n: "Erik Jenner", a: "", t: ["safety", "theory"], c: [L("Evidence of learned look-ahead in a chess network (2024)", "https://arxiv.org/abs/2406.00877")] },
    { n: "Jesse Hoogland & Daniel Murfet", a: "Timaeus", t: ["theory"], c: ["Developmental interpretability agenda (2023)", "Local learning coefficient & stagewise development — see Ch. 10"] },
    { n: "Andy Zou & Dan Hendrycks", a: "", t: ["probing", "theory"], c: [L("Representation Engineering (2023)", "https://arxiv.org/abs/2310.01405"), L("The Misguided Quest for Mechanistic AI Interpretability (2025, Hendrycks & Hiscott)", "https://ai-frontiers.org/articles/the-misguided-quest-for-mechanistic-ai-interpretability")] },
    { n: "Charbel-Raphaël Segerie", a: "", t: ["theory"], c: [L("Against Almost Every Theory of Impact of Interpretability (2023)", "https://www.alignmentforum.org/posts/LNA8mubrByG7SFacm/against-almost-every-theory-of-impact-of-interpretability")] },
    { n: "Kit Fraser-Taliente, Subhash Kantamneni & Euan Ong", a: "Anthropic", t: ["safety", "tools"], c: [L("Natural Language Autoencoders (2026, equal first authors)", "https://transformer-circuits.pub/2026/nla/index.html"), "NLAs used in Anthropic pre-deployment audits — see Ch. 10"] },
    { n: "Nicholas Sofroniew", a: "Anthropic", t: ["probing", "safety"], c: [L("Emotion Concepts and their Function (2026, first author)", "https://transformer-circuits.pub/2026/emotions/index.html"), L("Global workspace paper (2026, co-first author)", "https://transformer-circuits.pub/2026/workspace/index.html")] },
    { n: "Max McGuinness, Alex Serrano, Luke Bailey & Scott Emmons", a: "", t: ["probing", "safety"], c: [L("Neural Chameleons: models can learn to evade unseen activation monitors (2025)", "https://arxiv.org/abs/2512.11949")] },
    { n: "Abhay Sheshadri & Rowan Wang", a: "", t: ["safety"], c: [L("AuditBench (2026, first and senior authors)", "https://arxiv.org/abs/2602.22755"), L("Auditing agents (2025, Wang co-author)", "https://alignment.anthropic.com/2025/automated-auditing/")] },
    { n: "Dario Amodei", a: "Anthropic CEO", t: ["theory", "safety"], c: [L("The Urgency of Interpretability (2025)", "https://www.darioamodei.com/post/the-urgency-of-interpretability")] },
  ];
  const pdGrid = document.getElementById("pdGrid");
  if (pdGrid) {
    const bar = document.getElementById("pdBar"), search = document.getElementById("pdSearch"), count = document.getElementById("pdCount");
    let theme = "all";
    ["all", ...Object.keys(T)].forEach(k => {
      const b = E("button", { text: k === "all" ? "all" : T[k] }, bar);
      b.onclick = () => { theme = k; draw(); };
    });
    search.addEventListener("input", draw);
    function draw() {
      [...bar.children].forEach((b, i) => b.classList.toggle("on", (i === 0 ? "all" : Object.keys(T)[i - 1]) === theme));
      const q = search.value.toLowerCase().trim();
      const list = P.filter(p => (theme === "all" || p.t.includes(theme)) &&
        (!q || (p.n + " " + p.a + " " + p.c.join(" ")).toLowerCase().replace(/<[^>]+>/g, "").includes(q)));
      count.textContent = `${list.length} of ${P.length} entries`;
      pdGrid.innerHTML = list.map(p => `<div class="pd-card"><h4>${p.n}</h4>${p.a ? `<div class="aff">${p.a}</div>` : ""}<ul>${p.c.map(c => `<li>${c}</li>`).join("")}</ul><div class="tags">${p.t.map(t => `<span class="chip ${chipClass[t]}">${T[t]}</span>`).join("")}</div></div>`).join("")
        || `<p style="color:var(--ink-3)">No match. Try another word.</p>`;
    }
    draw();
  }

  /* ---------------- 3. Debate explorer ---------------- */
  const D = [
    { k: "Features real?", q: "Are 'features' the right basic unit for understanding a network?", yh: "Evidence for", nh: "Evidence against",
      y: ["Curve detectors, induction heads and many SAE features have clear meaning and clear causal effects.", "Linear directions work well for reading and steering many concepts (truth, refusal, sentiment).", "Similar features appear across different models (partial universality)."],
      n: ["Some features are curved manifolds or circles, not single directions.", "Feature splitting means the 'unit' depends on the dictionary size you pick.", "Computation may be spread in ways that no flat list of features captures (dark matter)."],
      now: "Most researchers keep features as a <b>working unit</b>, but now also study geometry (manifolds) and weights (parameter decomposition)." },
    { k: "Do SAEs find them?", q: "Do sparse autoencoders recover the model's true features?", yh: "For", nh: "Against",
      y: ["SAEs recover planted features in toy data.", "Millions of interpretable features on Claude 3 Sonnet and GPT-4; Golden Gate Claude shows causal effects.", "SAE features helped find the hidden goal in the auditing game and explained emergent misalignment."],
      n: ["Different seeds share only part of their features (~30% for a large Llama 3 8B SAE).", "SAEs on untrained transformers get similar interpretability scores.", "Linear probes beat SAE probes on a downstream harmful-intent task; DeepMind deprioritized basic SAE research (2025).", "Circuits in plain neurons can be as sparse as SAE circuits (Transluce, 2025)."],
      now: "Common middle view: SAEs are a <b>useful, lossy lens</b>, best for discovering unknowns, not the ground truth. In 2026 they remain a discovery tool in audits; probes and steering vectors do most known-concept tasks." },
    { k: "Useful for safety?", q: "Will interpretability make AI meaningfully safer?", yh: "Yes", nh: "Doubts",
      y: ["Auditing game: 3 of 4 blind teams found a hidden objective.", "White-box audits in system cards revealed evaluation awareness, including awareness the model did not say out loud (2026).", "Cheap probes work as production monitors at Google DeepMind and Anthropic (2026).", "An attribution graph ruled out a deception explanation in a real audit (Opus 4.6, 2026)."],
      n: ["Segerie (2023): most theories of impact are unlikely to pay off in time.", "Hendrycks & Hiscott (2025): bottom-up understanding of complex systems may not work.", "Nanda (2025): interpretability will not reliably find deceptive AI by itself.", "AuditBench (2026): white-box tools helped an auditing agent less than black-box tools."],
      now: "Broad agreement that interpretability is <b>one layer</b> of defense in depth, with the clearest value in auditing and monitoring. Evaluation awareness made the case stronger in 2026: behaviour tests alone can mislead." },
    { k: "Ambitious vs pragmatic", q: "Should the field aim for full understanding, or for practical tools?", yh: "Ambitious", nh: "Pragmatic",
      y: ["Only deep understanding can give strong guarantees about what a model will do.", "Attribution graphs and parameter decomposition show steady progress toward real mechanisms.", "Basic science compounds: universality gives footholds in future models."],
      n: ["'Near-complete reverse-engineering still feels far out of reach' (GDM, Dec 2025).", "Simple baselines (probes, steering vectors) often match fancy methods on real tasks.", "Proxy tasks give clear measures of progress; pure understanding is hard to measure."],
      now: "Both continue. GDM leans <b>pragmatic</b>; Anthropic works at both ends; Goodfire focuses on shaping training. Most groups now judge methods against simple baselines." },
    { k: "Can it scale?", q: "Can mechanistic interpretability keep up with ever larger models?", yh: "Hopeful", nh: "Worried",
      y: ["Automation: auditing agents and auto-interpretability scale with compute.", "New units (transcoders, CLTs) give graphs for real frontier models, including a production model (2026).", "Activation-to-text tools (NLAs, activation oracles) now run at scale in audits and on ~1-trillion-parameter models (Chapter 10)."],
      n: ["Attribution graphs give clear insight on only part of prompts, with large error nodes.", "Human analysis takes hours per prompt.", "Each new model may need new dictionaries trained at great cost."],
      now: "Scaling is <b>the</b> central open problem. Automation is the main bet, but AuditBench's 'tool-to-agent gap' shows that good tools do not automatically make good agents." },
    { k: "Can models hide?", q: "Will interpretability tools keep working if a model learns to hide its thoughts?", yh: "Reasons for hope", nh: "Worrying signs",
      y: ["The evasion in Neural Chameleons was trained on purpose; the paper does not show that current models do this on their own.", "Probe ensembles and non-linear probes resisted evasion better.", "White-box tools found test awareness that models did not say out loud, so internals still carried the signal (2026 system cards)."],
      n: ["Neural Chameleons (2025): models fine-tuned to dodge probes for harmless concepts also dodged unseen safety probes.", "Training directly against a monitor can teach a model to hide the monitored signal.", "Models already leave some thoughts out of their text, for example unspoken evaluation awareness."],
      now: "A <b>live risk</b>, not a demonstrated failure. Common advice: do not train directly against a monitor, and stress-test monitors against deliberate evasion." },
    { k: "Enumerative safety", q: "Could we list every feature and prove a model is safe?", yh: "Possible?", nh: "Unlikely",
      y: ["Larger dictionaries keep finding more features, including safety-relevant ones (deception, sycophancy).", "Dictionaries can be searched automatically for dangerous concepts."],
      n: ["No dictionary is complete: dark matter and a long tail of rare features.", "Features differ across seeds and sizes, so 'the' list is not well defined.", "A dangerous computation might not need a single named feature."],
      now: "Treated as a <b>long-term hope</b>, not a near-term plan." },
  ];
  const dbList = document.getElementById("dbList");
  if (dbList) {
    const show = i => {
      const d = D[i];
      [...dbList.children].forEach((b, j) => b.classList.toggle("on", j === i));
      document.getElementById("dbQ").textContent = d.q;
      document.getElementById("dbYesH").textContent = d.yh;
      document.getElementById("dbNoH").textContent = d.nh;
      document.getElementById("dbYes").innerHTML = d.y.map(x => `<li>${x}</li>`).join("");
      document.getElementById("dbNo").innerHTML = d.n.map(x => `<li>${x}</li>`).join("");
      document.getElementById("dbNow").innerHTML = `<b>State of play, 2026:</b> ${d.now}`;
    };
    D.forEach((d, i) => { const b = E("button", { class: "btn", text: d.k }, dbList); b.onclick = () => show(i); });
    show(0);
  }

  /* ---------------- 4. Dark matter schematic ---------------- */
  const dmCurve = document.getElementById("dmCurve");
  if (dmCurve) {
    const x0 = 60, x1 = 650, y0 = 210, top = 30;
    const f = t => y0 - (y0 - top) * Math.pow(1 + 40 * t, -1.1); // power-law-like long tail
    const pts = []; for (let i = 0; i <= 200; i++) { const t = i / 200; pts.push([x0 + t * (x1 - x0), f(t)]); }
    const path = p => p.map((q, i) => (i ? "L" : "M") + q[0].toFixed(1) + " " + q[1].toFixed(1)).join(" ");
    dmCurve.setAttribute("d", path(pts));
    const slider = document.getElementById("dmSize"), out = document.getElementById("dmOut");
    function upd() {
      const s = +slider.value; const cut = 0.04 + 0.5 * s * s + 0.1 * s; // grows slowly
      const xc = x0 + cut * (x1 - x0);
      const head = pts.filter(q => q[0] <= xc), tail = pts.filter(q => q[0] >= xc);
      document.getElementById("dmHead").setAttribute("d", path(head) + ` L ${xc} ${y0} L ${x0} ${y0} Z`);
      document.getElementById("dmTail").setAttribute("d", path(tail) + ` L ${x1} ${y0} L ${xc} ${y0} Z`);
      const c = document.getElementById("dmCut"); c.setAttribute("x1", xc); c.setAttribute("x2", xc);
      const lab = document.getElementById("dmCapLabel"); lab.setAttribute("x", Math.min(xc + 8, 470));
      const sizes = ["16K", "65K", "262K", "1M", "4M", "16M", "34M+"];
      out.textContent = sizes[Math.min(sizes.length - 1, Math.floor(s * sizes.length))] + " latents (schematic)";
    }
    slider.addEventListener("input", upd); upd();
  }

  /* ---------------- 5. Persona figure dots ---------------- */
  const pf = document.getElementById("pfDots");
  if (pf) {
    const r = MI.rng(7);
    for (let i = 0; i < 40; i++) S("circle", { cx: 110 + MI.gauss(r) * 22, cy: 185 + MI.gauss(r) * 12, r: 3.2, fill: "var(--blue)", opacity: .55 }, pf);
    for (let i = 0; i < 26; i++) S("circle", { cx: 320 + MI.gauss(r) * 22, cy: 112 + MI.gauss(r) * 12, r: 3.2, fill: "var(--red)", opacity: .55 }, pf);
  }

  /* ---------------- 6. Open problems map + checklist ---------------- */
  const OP = [
    { title: "METHODS & FOUNDATIONS", x: 10, w: 210, color: "var(--purple)", soft: "var(--purple-soft)", h: 50, items: [
      { label: "Better decomposition", d: "SAEs leave reconstruction error, split features arbitrarily, and miss non-linear features. We need units that capture the model's real computation.", ex: "Transcoders, cross-layer transcoders, parameter decomposition (APD, SPD), weight-sparse models." },
      { label: "Validate explanations", d: "How do we know an explanation is right? We need tests that cannot be fooled by illusions.", ex: "Causal scrubbing, interventions on attribution graphs, ground-truth testbeds such as Tracr, InterpBench and MIB; model organisms with known hidden behaviours (AuditBench, 2026)." },
      { label: "Attention & weights", d: "Most tools study activations. We still struggle to explain why heads attend where they do, and what each weight does.", ex: "QK attributions (2025), interference weights (2025–26), bilinear MLPs." },
      { label: "Non-linear geometry", d: "Some representations are circles or manifolds. Methods built on single directions miss them.", ex: "Circular day/month features; character-count manifolds (2025)." },
      { label: "Automation", d: "Human analysis does not scale. We need AI systems that do interpretability reliably.", ex: "Auto-interpretability, auditing agents, NLA investigator agents in system-card audits (2026), self-explaining models. Open issue: the 'tool-to-agent gap'." },
    ]},
    { title: "APPLICATIONS", x: 235, w: 210, color: "var(--accent)", soft: "var(--accent-soft)", h: 50, items: [
      { label: "Monitoring", d: "Detect bad behaviour or misuse in real time, cheaply and robustly.", ex: "Linear probes in production (Gemini, Constitutional Classifiers++, 2026), reward-hacking probes. Open issue: robustness to evasion (Neural Chameleons)." },
      { label: "Auditing", d: "Find hidden goals, backdoors or deception before release.", ex: "Auditing games, system-card white-box audits, finding unspoken evaluation awareness." },
      { label: "Control & editing", d: "Change behaviour precisely without retraining, and without side effects.", ex: "Steering vectors, ROME-style editing, concept erasure, probes as training rewards (2026)." },
      { label: "Prediction", d: "Predict how a model will behave in new situations, or what training will do.", ex: "Persona features predicting emergent misalignment; developmental interpretability." },
      { label: "Microscope AI", d: "Extract new scientific knowledge from models.", ex: "Biology models; AlphaZero chess concepts." },
    ]},
    { title: "SOCIO-TECHNICAL", x: 460, w: 210, color: "var(--green)", soft: "var(--green-soft)", h: 50, items: [
      { label: "Policy & governance", d: "How should regulators and labs use interpretability results? What counts as evidence?", ex: "Amodei's call for light-touch transparency rules (2025). In 2026, white-box audits appear in Anthropic system cards, but no common standard across labs exists." },
      { label: "Standards & benchmarks", d: "Shared ways to compare methods so that progress is real, not only claimed.", ex: "MIB, SAEBench, proxy tasks." },
      { label: "Dual use", d: "Better understanding can also increase capabilities, or help remove safety features (for example, refusal ablation).", ex: "'Abliteration' of open models with the refusal direction." },
      { label: "Communicating results", d: "Turn technical findings into claims that non-experts can trust and act on.", ex: "System cards, public explainers like this one." },
    ]},
  ];
  const opSvg = document.getElementById("opSvg");
  if (opSvg) boxMap(opSvg, document.getElementById("opDetail"), OP,
    it => `<h4>${it.label}</h4>${it.d}<div class="ev"><b>Examples of current work:</b> ${it.ex}</div>`);

  const opList = document.getElementById("opList");
  if (opList) {
    const bar = document.getElementById("opBar");
    let total = 0;
    OP.forEach(g => {
      const grp = E("div", { class: "op-group" }, opList);
      E("h4", { text: g.title.charAt(0) + g.title.slice(1).toLowerCase() }, grp);
      g.items.forEach(it => {
        total++;
        const row = E("div", { class: "op-item" }, grp);
        const cb = E("input", { type: "checkbox", "aria-label": "I can explain: " + it.label }, row);
        E("div", { html: `<b>${it.label}</b><div class="more">${it.d} <i>Examples: ${it.ex}</i></div>` }, row);
        row.addEventListener("click", e => { if (e.target !== cb) row.classList.toggle("open"); });
        cb.addEventListener("change", () => {
          const n = opList.querySelectorAll("input:checked").length;
          bar.style.width = (100 * n / total) + "%";
        });
      });
    });
  }
})();
