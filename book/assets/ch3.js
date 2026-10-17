/* Chapter 3 figures and widgets. Needs app.js (window.MI) and ch3-toy.js (ToyModel). */
(function () {
  const S = MI.svg;
  const FEAT_COLORS = ["var(--accent)", "var(--blue)", "var(--green)", "var(--purple)", "var(--yellow)", "var(--red)"];

  /* ---------------- static figure: top-activation bars (Fig 3.3) ---------------- */
  (function () {
    const g1 = document.getElementById("hist1"), g2 = document.getElementById("hist2");
    if (!g1) return;
    const heights = [150, 142, 135, 128, 122, 117, 111, 106, 100, 95, 90, 84, 79, 74];
    const poly = [0, 1, 0, 2, 1, 0, 1, 2, 0, 1, 0, 2, 1, 1];
    const pc = ["var(--accent)", "var(--blue)", "var(--purple)"];
    heights.forEach((h, i) => {
      S("rect", { x: 48 + i * 19.5, y: 200 - h, width: 15, height: h, rx: 2, fill: "var(--green)", opacity: 0.85 }, g1);
      S("rect", { x: 408 + i * 19.5, y: 200 - h, width: 15, height: h, rx: 2, fill: pc[poly[i]], opacity: 0.85 }, g2);
    });
  })();

  /* ---------------- static figure: toy architecture (Fig 3.5) ---------------- */
  (function () {
    const g = document.getElementById("toyArch");
    if (!g) return;
    const ins = [40, 70, 100, 130, 160], hid = [75, 125];
    ins.forEach(y => hid.forEach(h => {
      S("line", { x1: 92, y1: y, x2: 348, y2: h, stroke: "var(--ink-3)", "stroke-width": 1, opacity: 0.5 }, g);
      S("line", { x1: 372, y1: h, x2: 628, y2: y, stroke: "var(--ink-3)", "stroke-width": 1, opacity: 0.5 }, g);
    }));
    ins.forEach((y, i) => {
      S("circle", { cx: 80, cy: y, r: 11, fill: "var(--panel)", stroke: FEAT_COLORS[i], "stroke-width": 2.5 }, g);
      S("circle", { cx: 640, cy: y, r: 11, fill: "var(--panel)", stroke: FEAT_COLORS[i], "stroke-width": 2.5 }, g);
      S("text", { x: 80, y: y + 4, "text-anchor": "middle", "font-size": 11 }, g).textContent = i + 1;
      S("text", { x: 640, y: y + 4, "text-anchor": "middle", "font-size": 11 }, g).textContent = i + 1;
    });
    hid.forEach(h => S("circle", { cx: 360, cy: h, r: 14, fill: "var(--accent-soft)", stroke: "var(--accent)", "stroke-width": 2 }, g));
  })();

  /* ---------------- static figure: sticky fractions (Fig 3.6) ---------------- */
  (function () {
    const g = document.getElementById("stickyG");
    if (!g) return;
    const rows = [
      { f: 1, lab: "1", name: "dedicated dimension", shape: "one" },
      { f: 3 / 4, lab: "3/4", name: "tetrahedron (4 in 3D)", shape: "tet" },
      { f: 2 / 3, lab: "2/3", name: "triangle (3 in 2D)", shape: "tri" },
      { f: 1 / 2, lab: "1/2", name: "antipodal pair (2 in 1D)", shape: "di" },
      { f: 2 / 5, lab: "2/5", name: "pentagon (5 in 2D)", shape: "pent" },
      { f: 3 / 8, lab: "3/8", name: "square antiprism (8 in 3D)", shape: "anti" },
      { f: 0, lab: "0", name: "feature not learned", shape: "none" },
    ];
    S("text", { x: 300, y: 18, "font-size": 12, class: "muted" }, g).textContent = "fraction of a dimension per feature (D)";
    rows.forEach((r, i) => {
      const y = 40 + i * 37, cx = 40;
      // glyph
      const gg = S("g", { transform: `translate(${cx},${y})` }, g);
      const st = { fill: "none", stroke: "var(--accent)", "stroke-width": 1.8 };
      const poly = (k, rad, rot) => { const p = []; for (let j = 0; j < k; j++) { const a = rot + j * 2 * Math.PI / k; p.push([rad * Math.cos(a), rad * Math.sin(a)]); } return p; };
      const dots = pts => pts.forEach(p => S("circle", { cx: p[0], cy: p[1], r: 2.6, fill: "var(--accent)" }, gg));
      if (r.shape === "one") { S("line", { x1: 0, y1: 0, x2: 14, y2: 0, ...st }, gg); dots([[14, 0]]); }
      if (r.shape === "di") { S("line", { x1: -14, y1: 0, x2: 14, y2: 0, ...st }, gg); dots([[-14, 0], [14, 0]]); }
      if (r.shape === "tri") { const p = poly(3, 14, -Math.PI / 2); S("polygon", { points: p.map(q => q.join(",")).join(" "), ...st }, gg); dots(p); }
      if (r.shape === "pent") { const p = poly(5, 14, -Math.PI / 2); S("polygon", { points: p.map(q => q.join(",")).join(" "), ...st }, gg); dots(p); }
      if (r.shape === "tet") { const p = [[0, -15], [-13, 8], [13, 8], [2, 0]]; [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]].forEach(e => S("line", { x1: p[e[0]][0], y1: p[e[0]][1], x2: p[e[1]][0], y2: p[e[1]][1], ...st, "stroke-width": 1.3 }, gg)); dots(p); }
      if (r.shape === "anti") { const a = poly(4, 13, 0), b = poly(4, 13, Math.PI / 4); S("polygon", { points: a.map(q => q.join(",")).join(" "), ...st, "stroke-width": 1.3 }, gg); S("polygon", { points: b.map(q => q.join(",")).join(" "), ...st, "stroke-width": 1.3, "stroke-dasharray": "3 2" }, gg); dots(a.concat(b)); }
      if (r.shape === "none") { S("circle", { cx: 0, cy: 0, r: 4, fill: "none", stroke: "var(--ink-3)", "stroke-width": 1.5 }, gg); }
      S("text", { x: 76, y: y + 5, "font-size": 13 }, g).textContent = r.name;
      S("rect", { x: 300, y: y - 9, width: 340, height: 18, rx: 4, fill: "var(--bg-2)" }, g);
      if (r.f > 0) S("rect", { x: 300, y: y - 9, width: 340 * r.f, height: 18, rx: 4, fill: "var(--accent)", opacity: 0.8 }, g);
      S("text", { x: 652, y: y + 5, "font-size": 14, "font-weight": 700 }, g).textContent = r.lab;
    });
  })();

  /* ---------------- static figure: days circle (Fig 3.7) ---------------- */
  (function () {
    const g = document.getElementById("daysG");
    if (!g) return;
    const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
    const cx = 190, cy = 140, R = 100;
    S("circle", { cx, cy, r: R, fill: "none", stroke: "var(--rule)", "stroke-width": 2 }, g);
    const pos = i => { const a = -Math.PI / 2 + i * 2 * Math.PI / 7; return [cx + R * Math.cos(a), cy + R * Math.sin(a)]; };
    // arc Mon -> Wed
    const a0 = -Math.PI / 2 + 0.12, a1 = -Math.PI / 2 + 2 * 2 * Math.PI / 7 - 0.12, r2 = R + 22;
    S("path", { d: `M ${cx + r2 * Math.cos(a0)} ${cy + r2 * Math.sin(a0)} A ${r2} ${r2} 0 0 1 ${cx + r2 * Math.cos(a1)} ${cy + r2 * Math.sin(a1)}`, fill: "none", stroke: "var(--accent)", "stroke-width": 2.5, "marker-end": "url(#a37)" }, g);
    S("text", { x: cx + 95, y: cy - 98, "font-size": 14, "font-weight": 700, fill: "var(--accent)" }, g).textContent = "+2";
    days.forEach((d, i) => {
      const [x, y] = pos(i);
      const hl = i === 0 || i === 2;
      S("circle", { cx: x, cy: y, r: 8, fill: hl ? "var(--accent)" : "var(--panel)", stroke: hl ? "var(--accent)" : "var(--ink-3)", "stroke-width": 2 }, g);
      const a = -Math.PI / 2 + i * 2 * Math.PI / 7;
      S("text", { x: cx + (R - 30) * Math.cos(a), y: cy + (R - 30) * Math.sin(a) + 5, "text-anchor": "middle", "font-size": 13, "font-weight": hl ? 700 : 400 }, g).textContent = d;
    });
  })();

  /* ---------------- widget: almost-orthogonal vectors ---------------- */
  (function () {
    const svg = document.getElementById("orthoSvg");
    if (!svg) return;
    const DS = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000];
    const dIn = document.getElementById("orthoD"), nIn = document.getElementById("orthoN");
    let seed = 7;
    function run() {
      const d = DS[+dIn.value], N = +nIn.value;
      document.getElementById("orthoDo").textContent = d;
      document.getElementById("orthoNo").textContent = N;
      const r = MI.rng(seed + d * 13 + N);
      const V = [];
      for (let i = 0; i < N; i++) { const v = new Float64Array(d); let s = 0; for (let k = 0; k < d; k++) { v[k] = MI.gauss(r); s += v[k] * v[k]; } s = Math.sqrt(s); for (let k = 0; k < d; k++) v[k] /= s; V.push(v); }
      const B = 50, bins = new Array(B).fill(0);
      let sumAbs = 0, maxAbs = 0, cnt = 0;
      for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) {
        let c = 0; const a = V[i], b = V[j]; for (let k = 0; k < d; k++) c += a[k] * b[k];
        const idx = Math.min(B - 1, Math.floor((c + 1) / 2 * B)); bins[idx]++;
        sumAbs += Math.abs(c); maxAbs = Math.max(maxAbs, Math.abs(c)); cnt++;
      }
      svg.innerHTML = "";
      const x0 = 40, x1 = 620, y0 = 190, H = 160, mx = Math.max(...bins);
      S("line", { x1: x0, y1: y0, x2: x1, y2: y0, class: "rule", stroke: "var(--rule)" }, svg);
      bins.forEach((v, i) => {
        const w = (x1 - x0) / B, h = H * v / mx;
        S("rect", { x: x0 + i * w + 1, y: y0 - h, width: w - 2, height: h, fill: "var(--accent)", opacity: 0.8, rx: 1.5 }, svg);
      });
      [-1, -0.5, 0, 0.5, 1].forEach(t => {
        const x = x0 + (t + 1) / 2 * (x1 - x0);
        S("line", { x1: x, y1: y0, x2: x, y2: y0 + 5, stroke: "var(--ink-3)" }, svg);
        S("text", { x, y: y0 + 20, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = t;
      });
      S("text", { x: 330, y: 226, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "cosine similarity between pairs  (0 = perpendicular)";
      const ang = Math.acos(Math.min(1, maxAbs)) * 180 / Math.PI;
      document.getElementById("orthoOut").textContent =
        `d = ${d}, N = ${N}, pairs = ${cnt}.  mean |cos| = ${(sumAbs / cnt).toFixed(3)}  (theory ≈ 0.8/√d = ${(0.798 / Math.sqrt(d)).toFixed(3)}),  max |cos| = ${maxAbs.toFixed(3)}  → every pair is at least ${ang.toFixed(0)}° apart.`;
    }
    dIn.oninput = run; nIn.oninput = run;
    document.getElementById("orthoRe").onclick = () => { seed++; run(); };
    run();
  })();

  /* ---------------- widget: interference with 2 features in 1 dim ---------------- */
  (function () {
    const svg = document.getElementById("interfSvg");
    if (!svg) return;
    const st = { x1: 0.8, x2: 0, b: 0, relu: true };
    const bind = (id, key) => {
      const el = document.getElementById(id), o = document.getElementById(id + "o");
      el.oninput = () => { st[key] = +el.value; o.textContent = (+el.value).toFixed(2); draw(); };
    };
    bind("ifx1", "x1"); bind("ifx2", "x2"); bind("ifb", "b");
    document.getElementById("ifMode").onclick = e => {
      const m = e.target.dataset.m; if (!m) return;
      st.relu = m === "relu";
      [...e.currentTarget.children].forEach(b => b.classList.toggle("on", b.dataset.m === m));
      draw();
    };
    function bar(x, base, val, col, label, ghost) {
      const H = 120, y0 = 160, scale = H / 1.2;
      if (ghost != null) S("rect", { x, y: y0 - Math.max(0, ghost) * scale, width: 34, height: Math.abs(ghost) * scale, fill: "none", stroke: "var(--ink-3)", "stroke-dasharray": "4 3", rx: 3 }, svg);
      const h = val * scale;
      S("rect", { x: x + 4, y: h >= 0 ? y0 - h : y0, width: 26, height: Math.abs(h), fill: col, rx: 3, opacity: 0.9 }, svg);
      S("text", { x: x + 17, y: y0 + (h < 0 ? Math.abs(h) + 16 : 18), "text-anchor": "middle", "font-size": 12 }, svg).textContent = val.toFixed(2);
      S("text", { x: x + 17, y: 200, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = label;
    }
    function draw() {
      svg.innerHTML = "";
      const h = st.x1 - st.x2;
      const p1 = h + st.b, p2 = -h + st.b;
      const o1 = st.relu ? Math.max(0, p1) : p1, o2 = st.relu ? Math.max(0, p2) : p2;
      S("line", { x1: 20, y1: 160, x2: 160, y2: 160, stroke: "var(--rule)" }, svg);
      S("line", { x1: 470, y1: 160, x2: 620, y2: 160, stroke: "var(--rule)" }, svg);
      S("text", { x: 90, y: 20, "text-anchor": "middle", "font-size": 13, "font-weight": 700 }, svg).textContent = "input x";
      bar(40, 0, st.x1, "var(--accent)", "x₁");
      bar(100, 0, st.x2, "var(--blue)", "x₂");
      // number line
      const cx = 315, sc = 90;
      S("text", { x: cx, y: 20, "text-anchor": "middle", "font-size": 13, "font-weight": 700 }, svg).textContent = "hidden h = x₁ − x₂";
      S("line", { x1: cx - sc - 10, y1: 90, x2: cx + sc + 10, y2: 90, stroke: "var(--ink-3)", "stroke-width": 1.5 }, svg);
      [-1, 0, 1].forEach(t => { S("line", { x1: cx + t * sc, y1: 84, x2: cx + t * sc, y2: 96, stroke: "var(--ink-3)" }, svg); S("text", { x: cx + t * sc, y: 112, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = t; });
      S("text", { x: cx + sc, y: 72, "text-anchor": "middle", "font-size": 11, fill: "var(--accent)" }, svg).textContent = "W₁";
      S("text", { x: cx - sc, y: 72, "text-anchor": "middle", "font-size": 11, fill: "var(--blue)" }, svg).textContent = "W₂";
      S("circle", { cx: cx + h * sc, cy: 90, r: 8, fill: "var(--ink)" }, svg);
      S("text", { x: cx, y: 145, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = `h = ${h.toFixed(2)}`;
      S("text", { x: 545, y: 20, "text-anchor": "middle", "font-size": 13, "font-weight": 700 }, svg).textContent = st.relu ? "output (ReLU)" : "output (linear)";
      bar(490, 0, o1, "var(--accent)", "x′₁", st.x1);
      bar(560, 0, o2, "var(--blue)", "x′₂", st.x2);
      const err = (o1 - st.x1) ** 2 + (o2 - st.x2) ** 2;
      let msg = `x′₁ = ${st.relu ? "ReLU" : ""}(${h.toFixed(2)} + ${st.b.toFixed(2)}) = ${o1.toFixed(2)}   x′₂ = ${st.relu ? "ReLU" : ""}(${(-h).toFixed(2)} + ${st.b.toFixed(2)}) = ${o2.toFixed(2)}   squared error = ${err.toFixed(3)}`;
      if (st.x1 > 0.05 && st.x2 > 0.05) msg += "   ← both on: they cancel inside h. This is the cost of superposition.";
      else if (!st.relu && err > 0.01) msg += "   ← negative interference is not filtered without ReLU.";
      else if (err < 0.005) msg += "   ✓ recovered";
      document.getElementById("interfOut").textContent = msg;
      // dashed boxes legend
      S("text", { x: 545, y: 40, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = "dashed = true value";
    }
    draw();
    MI.onTheme(draw);
  })();

  /* ---------------- widget: live toy model ---------------- */
  (function () {
    const svg = document.getElementById("toySvg");
    if (!svg || !window.ToyModel) return;
    const gram = document.getElementById("gramSvg"), bars = document.getElementById("barSvg");
    const sIn = document.getElementById("tS"), rIn = document.getElementById("tR");
    const N = 5, M = 2, MAX = 4000, PER_FRAME = 20;
    let relu = true, model = null, running = false, seed = 3;

    function make() {
      model = new ToyModel({ n: N, m: M, S: +sIn.value, decay: +rIn.value, relu, seed, lr: 0.01, batch: 256 });
    }
    function stop() { running = false; document.getElementById("tTrain").textContent = "▶ Train"; }
    function resetAll(newSeed) { stop(); if (newSeed) seed++; make(); draw(); document.getElementById("tMsg").textContent = "Press Train."; }

    sIn.oninput = () => { document.getElementById("tSo").textContent = (+sIn.value).toFixed(2); resetAll(false); };
    rIn.oninput = () => { document.getElementById("tRo").textContent = (+rIn.value).toFixed(2); resetAll(false); };
    document.getElementById("tMode").onclick = e => {
      const m = e.target.dataset.m; if (!m) return;
      relu = m === "relu";
      [...e.currentTarget.children].forEach(b => b.classList.toggle("on", b.dataset.m === m));
      resetAll(false);
    };
    document.querySelectorAll("#toyWidget [data-p]").forEach(b => b.onclick = () => {
      sIn.value = b.dataset.p; document.getElementById("tSo").textContent = (+sIn.value).toFixed(2);
      resetAll(false); start();
    });
    document.getElementById("tReset").onclick = () => resetAll(true);
    document.getElementById("tTrain").onclick = () => (running ? stop() : start());

    function start() {
      if (!model || model.t >= MAX) make();
      running = true; document.getElementById("tTrain").textContent = "❚❚ Pause";
      document.getElementById("tMsg").textContent = "Training… watch the arrows move.";
      requestAnimationFrame(loop);
    }
    function loop() {
      if (!running) return;
      for (let k = 0; k < PER_FRAME && model.t < MAX; k++) model.step();
      draw();
      if (model.t >= MAX) { stop(); describe(); return; }
      requestAnimationFrame(loop);
    }
    function describe() {
      const cols = [...Array(N).keys()].map(i => model.col(i));
      const norms = cols.map(c => Math.hypot(c[0], c[1]));
      const stored = norms.filter(v => v > 0.5).length;
      const meanB = Array.from(model.b).filter((_, i) => norms[i] > 0.5).reduce((a, v) => a + v, 0) / Math.max(1, stored);
      let txt = `Done. ${stored} of ${N} features stored. `;
      if (stored <= 2) txt += "No superposition: the model keeps only the most important features, at 90° (like PCA).";
      else if (stored === 4) txt += "Two antipodal pairs: each pair shares one dimension, pointing in opposite directions.";
      else if (stored === 5) txt += `All five stored, about 72° apart: a pentagon. Mean bias = ${meanB.toFixed(2)} (negative: it cuts positive interference).`;
      else txt += "Partial superposition.";
      document.getElementById("tMsg").textContent = txt;
    }

    function draw() {
      svg.innerHTML = "";
      const c = 180, sc = 115;
      S("circle", { cx: c, cy: c, r: sc, fill: "none", stroke: "var(--rule)", "stroke-dasharray": "4 4" }, svg);
      S("line", { x1: 20, y1: c, x2: 340, y2: c, stroke: "var(--rule)" }, svg);
      S("line", { x1: c, y1: 20, x2: c, y2: 340, stroke: "var(--rule)" }, svg);
      S("text", { x: 338, y: c - 6, "text-anchor": "end", "font-size": 11, class: "muted" }, svg).textContent = "hidden dim 1";
      S("text", { x: c + 6, y: 30, "font-size": 11, class: "muted" }, svg).textContent = "hidden dim 2";
      S("text", { x: c + sc * 0.72, y: c + sc * 0.72 + 14, "font-size": 10, class: "muted" }, svg).textContent = "‖W‖ = 1";
      if (!model) return;
      for (let i = 0; i < N; i++) {
        const w = model.col(i), nr = Math.hypot(w[0], w[1]);
        const x = c + w[0] * sc, y = c - w[1] * sc;
        const col = FEAT_COLORS[i];
        S("line", { x1: c, y1: c, x2: x, y2: y, stroke: col, "stroke-width": 4, "stroke-linecap": "round", opacity: nr < 0.15 ? 0.3 : 1 }, svg);
        S("circle", { cx: x, cy: y, r: 6, fill: col }, svg);
        if (nr > 0.15) {
          const lx = c + w[0] * sc * (1 + 16 / (nr * sc)), ly = c - w[1] * sc * (1 + 16 / (nr * sc));
          S("text", { x: lx, y: ly + 4, "text-anchor": "middle", "font-size": 13, "font-weight": 700, fill: col }, svg).textContent = i + 1;
        }
      }
      // gram
      gram.innerHTML = "";
      const G = model.gram(), cs = 30, ox = 40, oy = 30;
      for (let i = 0; i < N; i++) {
        S("text", { x: ox - 10, y: oy + i * cs + cs / 2 + 4, "text-anchor": "middle", "font-size": 11, fill: FEAT_COLORS[i], "font-weight": 700 }, gram).textContent = i + 1;
        S("text", { x: ox + i * cs + cs / 2, y: oy - 8, "text-anchor": "middle", "font-size": 11, fill: FEAT_COLORS[i], "font-weight": 700 }, gram).textContent = i + 1;
        for (let j = 0; j < N; j++) {
          S("rect", { x: ox + j * cs, y: oy + i * cs, width: cs - 2, height: cs - 2, rx: 3, fill: MI.diverge(G[i][j] / 1.2) }, gram);
          S("text", { x: ox + j * cs + cs / 2 - 1, y: oy + i * cs + cs / 2 + 3, "text-anchor": "middle", "font-size": 8.5, fill: Math.abs(G[i][j]) > 0.6 ? "#fff" : "var(--ink-3)" }, gram).textContent = G[i][j].toFixed(1).replace("-0.0", "0.0");
        }
      }
      S("text", { x: ox + 75, y: oy + N * cs + 16, "font-size": 10.5, class: "muted", "text-anchor": "middle" }, gram).textContent = "orange = positive overlap · blue = negative";
      // bars
      bars.innerHTML = "";
      const bw = 34;
      S("line", { x1: 10, y1: 60, x2: 210, y2: 60, stroke: "var(--rule)" }, bars);
      for (let i = 0; i < N; i++) {
        const w = model.col(i), nr = Math.hypot(w[0], w[1]), b = model.b[i];
        const x = 20 + i * 40;
        S("rect", { x, y: 60 - Math.min(1.3, nr) * 40, width: bw / 2 - 1, height: Math.min(1.3, nr) * 40, fill: FEAT_COLORS[i], rx: 2 }, bars);
        const bh = Math.max(-0.5, Math.min(0.5, b)) * 80;
        S("rect", { x: x + bw / 2, y: bh >= 0 ? 60 - bh : 60, width: bw / 2 - 1, height: Math.abs(bh), fill: "var(--ink-3)", rx: 2 }, bars);
        S("text", { x: x + bw / 2, y: 118, "text-anchor": "middle", "font-size": 11, fill: FEAT_COLORS[i], "font-weight": 700 }, bars).textContent = i + 1;
      }
      S("text", { x: 210, y: 14, "text-anchor": "end", "font-size": 10, class: "muted" }, bars).textContent = "colored = ‖Wᵢ‖, grey = bias bᵢ";
      // stats
      let fro = 0; for (let i = 0; i < N; i++) fro += G[i][i];
      document.getElementById("tStep").textContent = model.t;
      document.getElementById("tLoss").textContent = isNaN(model.loss) ? "–" : model.loss.toFixed(4);
      document.getElementById("tFro").textContent = fro.toFixed(2);
      document.getElementById("tDstar").textContent = fro > 0.01 ? (M / fro).toFixed(2) : "–";
    }
    make(); draw();
    MI.onTheme(draw);
    const auto = location.hash.match(/^#train(\d+)$/);
    if (auto) { const b = document.querySelector(`#toyWidget [data-p="0.${auto[1]}"]`) || document.querySelector('#toyWidget [data-p="0"]'); b.click(); }
  })();

  /* ---------------- widget: phase diagram explorer ---------------- */
  (function () {
    const svg = document.getElementById("phaseSvg");
    if (!svg) return;
    // Rows: density 10^(-2j/10), j=0..10 (top = dense). Cols: importance ratio 10^(-1+2i/12).
    // N = extra feature ignored, D = extra feature takes the dimension, S = superposition.
    const GRID = ["NNNNNNNDDDDDD", "NNNNNNSDDDDDD", "NNNNSSSSSDDDD", "NNSSSSSSSSSDD", "SSSSSSSSSSSSS", "SSSSSSSSSSSSS", "SSSSSSSSSSSSS", "SSSSSSSSSSSSS", "SSSSSSSSSSSSS", "SSSSSSSSSSSSS", "SSSSSSSSSSSSS"];
    const rows = GRID.length, cols = GRID[0].length;
    const x0 = 70, y0 = 20, W = 330, H = 260, cw = W / cols, ch = H / rows;
    const fill = { N: "var(--bg-2)", D: "var(--blue)", S: "var(--accent)" };
    const info = document.getElementById("phaseInfo");
    const text = {
      N: ["Extra feature ignored", "The model stores only feature 1: W = [1, 0]. The extra feature is not important enough to pay for, and it is too dense to share."],
      D: ["Extra feature gets the dimension", "The extra feature is more important and features are dense, so it takes the only dimension: W = [0, 1]. Feature 1 is dropped."],
      S: ["Superposition", "Both features are stored as an antipodal pair: W = [1, −1]. They are sparse, so they are rarely on together, and the ReLU filters the negative interference."],
    };
    function draw(sel) {
      svg.innerHTML = "";
      for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
        const k = GRID[j][i];
        const r = S("rect", { x: x0 + i * cw, y: y0 + j * ch, width: cw - 1, height: ch - 1, fill: fill[k], opacity: k === "N" ? 1 : 0.8, stroke: sel && sel[0] === i && sel[1] === j ? "var(--ink)" : "none", "stroke-width": 2 }, svg);
        r.addEventListener("mouseenter", () => show(i, j));
        r.addEventListener("click", () => show(i, j));
      }
      S("text", { x: x0 + W / 2, y: y0 + H + 34, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "importance of extra feature (relative to feature 1) →";
      [[0, "0.1"], [6, "1"], [12, "10"]].forEach(([i, l]) => S("text", { x: x0 + (i + 0.5) * cw, y: y0 + H + 16, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = l);
      [[0, "1.0"], [5, "0.1"], [10, "0.01"]].forEach(([j, l]) => S("text", { x: x0 - 8, y: y0 + (j + 0.5) * ch + 4, "text-anchor": "end", "font-size": 11, class: "muted" }, svg).textContent = l);
      const t = S("text", { x: 16, y: y0 + H / 2, "text-anchor": "middle", "font-size": 12, class: "muted", transform: `rotate(-90 16 ${y0 + H / 2})` }, svg);
      t.textContent = "density 1 − S  (log scale)";
    }
    function show(i, j) {
      const k = GRID[j][i];
      const ratio = Math.pow(10, -1 + 2 * i / (cols - 1)), dens = Math.pow(10, -2 * j / (rows - 1));
      info.innerHTML = `<b>${text[k][0]}</b><br><span style="color:var(--ink-3);font-size:13px">importance ratio ≈ ${ratio.toFixed(2)}, density ≈ ${dens.toFixed(3)} (on ${(dens * 100).toFixed(1)}% of the time)</span><br>${text[k][1]}`;
      draw([i, j]);
    }
    draw();
    MI.onTheme(() => draw());
  })();
  /* ---------------- widget: linear representation (features as arrows that add) ---------------- */
  (function () {
    const svg = document.getElementById("lrhSvg");
    if (!svg) return;
    const st = { a: 1, b: 0, t: 90 };
    const ids = { a: "lrhA", b: "lrhB", t: "lrhT" };
    const setUI = () => {
      for (const k in ids) { document.getElementById(ids[k]).value = st[k]; document.getElementById(ids[k] + "o").textContent = k === "t" ? st.t + "°" : st[k].toFixed(2); }
    };
    for (const k in ids) document.getElementById(ids[k]).oninput = e => { st[k] = +e.target.value; setUI(); draw(); };
    document.getElementById("lrhPresets").onclick = e => {
      const d = e.target.dataset; if (d.a == null) return;
      st.a = +d.a; st.b = +d.b; setUI(); draw();
    };
    const arrow = (x1, y1, x2, y2, col, w, dash) => {
      const L = Math.hypot(x2 - x1, y2 - y1); if (L < 1) return;
      S("line", { x1, y1, x2, y2, stroke: col, "stroke-width": w, "stroke-dasharray": dash || null }, svg);
      const ux = (x2 - x1) / L, uy = (y2 - y1) / L, hs = 9;
      S("path", { d: `M${x2},${y2} L${x2 - hs * ux - hs * 0.5 * uy},${y2 - hs * uy + hs * 0.5 * ux} L${x2 - hs * ux + hs * 0.5 * uy},${y2 - hs * uy - hs * 0.5 * ux} z`, fill: col }, svg);
    };
    const txt = (x, y, t, o) => { const e = S("text", Object.assign({ x, y, "font-size": 12 }, o || {}), svg); e.textContent = t; return e; };
    function draw() {
      svg.innerHTML = "";
      const th = st.t * Math.PI / 180, ox = 70, oy = 250, sc = 110;
      const dR = [1, 0], dF = [Math.cos(th), Math.sin(th)];
      const P = (a, b) => [ox + sc * (a * dR[0] + b * dF[0]), oy - sc * (a * dR[1] + b * dF[1])];
      // faint lattice lines along both feature directions
      for (let i = 0; i <= 3; i++) {
        const v = i * 0.5;
        let [x1, y1] = P(v, 0), [x2, y2] = P(v, 1.6); S("line", { x1, y1, x2, y2, stroke: "var(--rule)" }, svg);
        [x1, y1] = P(0, v); [x2, y2] = P(1.6, v); S("line", { x1, y1, x2, y2, stroke: "var(--rule)" }, svg);
      }
      // reference words
      [["man", 0, 0], ["king", 1, 0], ["woman", 0, 1], ["queen", 1, 1]].forEach(([w, a, b]) => {
        const [x, y] = P(a, b);
        S("circle", { cx: x, cy: y, r: 4, fill: "var(--ink-3)" }, svg);
        txt(x + 6, y - 7, w, { class: "muted", "font-style": "italic" });
      });
      const [ur, vr] = P(1, 0), [uf, vf] = P(0, 1);
      arrow(ox, oy, ur, vr, "var(--accent)", 1.5, "3 3");
      arrow(ox, oy, uf, vf, "var(--blue)", 1.5, "3 3");
      const [xa, ya] = P(st.a, 0), [xh, yh] = P(st.a, st.b);
      arrow(ox, oy, xa, ya, "var(--accent)", 4);
      arrow(xa, ya, xh, yh, "var(--blue)", 4);
      arrow(ox, oy, xh, yh, "var(--ink)", 2.5, "6 4");
      // unit feature arrows for reference
      txt((ox + ur) / 2, oy + 24, "royal arrow", { fill: "var(--accent)", "font-weight": 700, "text-anchor": "middle" });
      txt(uf + 6, (oy + vf) / 2 + 20, "female", { fill: "var(--blue)", "font-weight": 700 }); txt(uf + 6, (oy + vf) / 2 + 34, "arrow", { fill: "var(--blue)", "font-weight": 700 });
      // probe readings (dot products)
      const h = [st.a * dR[0] + st.b * dF[0], st.a * dR[1] + st.b * dF[1]];
      const rR = h[0] * dR[0] + h[1] * dR[1], rF = h[0] * dF[0] + h[1] * dF[1];
      // projection of h onto the royal axis
      const [px, py] = P(rR, 0);
      S("line", { x1: xh, y1: yh, x2: px, y2: py, stroke: "var(--accent)", "stroke-dasharray": "2 3" }, svg);
      S("circle", { cx: px, cy: py, r: 3.5, fill: "var(--accent)" }, svg);
      // nearest word
      const words = [["man", 0, 0], ["king", 1, 0], ["woman", 0, 1], ["queen", 1, 1]];
      let best = words[0], bd = 1e9;
      words.forEach(w => { const d = (w[1] - st.a) ** 2 + (w[2] - st.b) ** 2; if (d < bd) { bd = d; best = w; } });
      txt(xh + 8, yh + 16, (bd < 0.01 ? "= " : "≈ ") + best[0], { "font-weight": 700 });
      // right panel: probes
      const bx = 420, y0 = 230, bs = 110;
      S("line", { x1: bx - 10, y1: y0, x2: 630, y2: y0, stroke: "var(--rule)" }, svg);
      txt(525, 24, "linear probes (one dot product each)", { "text-anchor": "middle", "font-weight": 700, "font-size": 13 });
      const probe = (x, truth, read, col, label) => {
        S("rect", { x, y: y0 - truth * bs, width: 50, height: truth * bs, fill: "none", stroke: "var(--ink-3)", "stroke-dasharray": "4 3", rx: 3 }, svg);
        S("rect", { x: x + 6, y: y0 - read * bs, width: 38, height: read * bs, fill: col, rx: 3, opacity: 0.9 }, svg);
        txt(x + 25, y0 - read * bs - 6, read.toFixed(2), { "text-anchor": "middle" });
        txt(x + 25, y0 + 18, label, { "text-anchor": "middle", class: "muted" });
        txt(x + 25, y0 + 34, "added: " + truth.toFixed(2), { "text-anchor": "middle", class: "muted", "font-size": 11 });
      };
      probe(440, st.a, rR, "var(--accent)", "royal?");
      probe(540, st.b, rF, "var(--blue)", "female?");
      txt(525, 44, "dashed = amount actually added", { "text-anchor": "middle", class: "muted", "font-size": 11 });
      const leak = Math.abs(rR - st.a) + Math.abs(rF - st.b);
      let msg = `activation = ${st.a.toFixed(2)}·royal + ${st.b.toFixed(2)}·female\nroyal probe = activation · royal = ${rR.toFixed(2)}
female probe = activation · female = ${rF.toFixed(2)}`;
      if (leak > 0.02) msg += `\n← the arrows are ${st.t}° apart, so cos = ${Math.cos(th).toFixed(2)}: each probe picks up ${Math.cos(th).toFixed(2)} × the other feature. This is interference.`;
      else msg += "\n✓ perpendicular arrows: each probe reads back exactly what was added.";
      document.getElementById("lrhOut").textContent = msg;
    }
    draw();
    MI.onTheme(draw);
  })();

  /* ---------------- widget: five features in 2D, linear vs ReLU readout ---------------- */
  (function () {
    const svg = document.getElementById("reluSvg");
    if (!svg) return;
    const W = [0, 1, 2, 3, 4].map(k => [Math.cos(Math.PI / 2 + k * 2 * Math.PI / 5), Math.sin(Math.PI / 2 + k * 2 * Math.PI / 5)]);
    const st = { on: [true, false, false, false, false], b: 0, relu: true };
    document.getElementById("reluFeats").onclick = e => {
      const k = e.target.dataset.k; if (k == null) return;
      st.on[k] = !st.on[k]; e.target.classList.toggle("on", st.on[k]); draw();
    };
    const bEl = document.getElementById("reluB");
    bEl.oninput = () => { st.b = +bEl.value; document.getElementById("reluBo").textContent = st.b.toFixed(2); draw(); };
    document.getElementById("reluMode").onclick = e => {
      const m = e.target.dataset.m; if (!m) return;
      st.relu = m === "relu";
      [...e.currentTarget.children].forEach(b => b.classList.toggle("on", b.dataset.m === m));
      draw();
    };
    const txt = (x, y, t, o) => { const e = S("text", Object.assign({ x, y, "font-size": 12 }, o || {}), svg); e.textContent = t; return e; };
    function draw() {
      svg.innerHTML = "";
      const cx = 125, cy = 135, sc = 80;
      S("circle", { cx, cy, r: sc, fill: "none", stroke: "var(--rule)" }, svg);
      W.forEach((w, k) => {
        const x = cx + sc * w[0], y = cy - sc * w[1], col = FEAT_COLORS[k];
        S("line", { x1: cx, y1: cy, x2: x, y2: y, stroke: col, "stroke-width": st.on[k] ? 4 : 2, opacity: st.on[k] ? 1 : 0.55 }, svg);
        S("circle", { cx: x, cy: y, r: 4, fill: col }, svg);
        txt(cx + (sc + 16) * w[0], cy - (sc + 16) * w[1] + 4, "f" + "₁₂₃₄₅"[k], { "text-anchor": "middle", fill: col, "font-weight": 700 });
      });
      const x = st.on.map(o => (o ? 1 : 0));
      const h = [0, 0]; W.forEach((w, k) => { h[0] += x[k] * w[0]; h[1] += x[k] * w[1]; });
      const hx = cx + sc * h[0], hy = cy - sc * h[1];
      S("line", { x1: cx, y1: cy, x2: hx, y2: hy, stroke: "var(--ink)", "stroke-width": 2.5, "stroke-dasharray": "6 4" }, svg);
      S("circle", { cx: hx, cy: hy, r: 6, fill: "var(--ink)" }, svg);
      txt(hx + 9, hy - 7, "h", { "font-weight": 700, "font-style": "italic" });
      txt(cx, 18, "hidden space (2 dims)", { "text-anchor": "middle", "font-weight": 700, "font-size": 13 });
      // right: readouts
      const y0 = 150, bs = 75, x0 = 300, step = 66;
      txt(470, 18, st.relu ? "readout: ReLU(fⱼ · h + b)" : "readout: fⱼ · h + b  (linear)", { "text-anchor": "middle", "font-weight": 700, "font-size": 13 });
      S("line", { x1: x0 - 10, y1: y0, x2: 630, y2: y0, stroke: "var(--ink-3)" }, svg);
      if (st.b !== 0) {
        S("line", { x1: x0 - 10, y1: y0 - st.b * bs, x2: 630, y2: y0 - st.b * bs, stroke: "var(--ink-3)", "stroke-dasharray": "2 4" }, svg);
      }
      let err = 0;
      W.forEach((w, j) => {
        const pre = w[0] * h[0] + w[1] * h[1] + st.b, out = st.relu ? Math.max(0, pre) : pre, col = FEAT_COLORS[j];
        err += (out - x[j]) ** 2;
        const bx = x0 + j * step;
        if (x[j]) S("rect", { x: bx, y: y0 - bs, width: 44, height: bs, fill: "none", stroke: "var(--ink-3)", "stroke-dasharray": "4 3", rx: 3 }, svg);
        S("rect", { x: bx + 2, y: pre >= 0 ? y0 - pre * bs : y0, width: 40, height: Math.abs(pre) * bs, fill: col, opacity: 0.22, rx: 3 }, svg);
        S("rect", { x: bx + 9, y: out >= 0 ? y0 - out * bs : y0, width: 26, height: Math.abs(out) * bs, fill: col, opacity: 0.95, rx: 3 }, svg);
        const wrong = Math.abs(out - x[j]) > 0.02;
        txt(bx + 22, 245, out.toFixed(2), { "text-anchor": "middle", "font-weight": 700, fill: wrong ? "var(--red)" : "var(--green)" });
        txt(bx + 22, 228, "f" + "₁₂₃₄₅"[j], { "text-anchor": "middle", class: "muted" });
      });
      const nOn = x.reduce((a, c) => a + c, 0);
      const offMax = Math.max(...W.map((w, j) => (x[j] ? -9 : w[0] * h[0] + w[1] * h[1])));
      let msg = `${nOn} feature${nOn === 1 ? "" : "s"} on · squared error = ${err.toFixed(3)}`;
      if (nOn === 0) msg += "\nTurn on a feature.";
      else if (!st.relu) msg += "\nLinear: off features show their raw interference (±). A bias moves all bars together, so it cannot fix this.";
      else if (nOn === 1 && st.b > -0.3) msg += "\nReLU has erased the negative readings. The +0.31 readings remain. Lower the bias below −0.31.";
      else if (nOn === 1) msg += `\n✓ All interference removed. The on feature reads ${(1 + st.b).toFixed(2)}: a little low, the price of the bias.`;
      else if (offMax + st.b > 0.01) msg += `\nInterference stacks up: an off feature now reads ${offMax.toFixed(2)} before bias. The ReLU can't remove it without also killing real signal. Collisions are costly.`;
      else msg += "\nLucky: these features are far enough apart that the interference stays negative.";
      document.getElementById("reluOut").textContent = msg;
    }
    draw();
    MI.onTheme(draw);
  })();

  /* ---------------- widget: sparsity vs loss (pentagon vs keep-2) ---------------- */
  (function () {
    const svg = document.getElementById("sparseSvg");
    if (!svg) return;
    const W = [0, 1, 2, 3, 4].map(k => [Math.cos(Math.PI / 2 + k * 2 * Math.PI / 5), Math.sin(Math.PI / 2 + k * 2 * Math.PI / 5)]);
    const G = W.map(a => W.map(c => a[0] * c[0] + a[1] * c[1]));
    let seed = 7;
    const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    // Monte Carlo once: loss as fraction of total signal 5·E[x²] = 5p/3
    const N = 1200, U = [...Array(N)].map(() => [...Array(5)].map(() => [rnd(), rnd()]));
    const Ss = [...Array(25)].map((_, i) => i * 0.04);
    const curve = Ss.map(S => {
      const p = 1 - S;
      let best = 1e9, bb = 0;
      for (let b = -0.6; b <= 0.61; b += 0.05) {
        let L = 0;
        for (const z of U) {
          const x = z.map(([u, v]) => (u < p ? v : 0));
          for (let j = 0; j < 5; j++) {
            let pre = b; for (let k = 0; k < 5; k++) pre += G[j][k] * x[k];
            const d = Math.max(0, pre) - x[j]; L += d * d;
          }
        }
        L /= N; if (L < best) { best = L; bb = b; }
      }
      // keep 2 perfect, predict the mean p/2 for the 3 dropped: 3·Var = 3(p/3 − p²/4)
      return { S, sup: best / (5 * p / 3), pca: 0.6 * (1 - 0.75 * p), b: bb };
    });
    const st = { S: 0.2 };
    const sEl = document.getElementById("spS");
    sEl.oninput = () => { st.S = +sEl.value; document.getElementById("spSo").textContent = st.S.toFixed(2); draw(); };
    let sampleSeed = 99;
    document.getElementById("spRe").onclick = () => { sampleSeed += 17; draw(); };
    const txt = (x, y, t, o) => { const e = S("text", Object.assign({ x, y, "font-size": 12 }, o || {}), svg); e.textContent = t; return e; };
    const interp = (key) => {
      const i = Math.min(Ss.length - 2, Math.floor(st.S / 0.04)), f = (st.S - Ss[i]) / 0.04;
      return curve[i][key] * (1 - f) + curve[i + 1][key] * f;
    };
    function draw() {
      svg.innerHTML = "";
      // left: sample strip
      let s2 = sampleSeed;
      const r2 = () => (s2 = (s2 * 48271) % 2147483647) / 2147483647;
      const p = 1 - st.S, cols = 40, x0 = 22, cw = 7.2, y0 = 50, rh = 26;
      txt(x0 + cols * cw / 2, 20, "40 random inputs (columns)", { "text-anchor": "middle", "font-weight": 700, "font-size": 13 });
      let coll = 0;
      for (let c = 0; c < cols; c++) {
        const on = [0, 1, 2, 3, 4].map(() => r2() < p);
        const n = on.filter(Boolean).length;
        if (n >= 2) { coll++; S("rect", { x: x0 + c * cw - cw / 2 + 0.5, y: y0 - 14, width: cw - 1, height: rh * 5 + 4, fill: "var(--red)", opacity: 0.16, rx: 2 }, svg); }
        on.forEach((o, k) => S("circle", { cx: x0 + c * cw, cy: y0 + k * rh, r: o ? 3 : 1.2, fill: o ? FEAT_COLORS[k] : "var(--ink-3)", opacity: o ? 1 : 0.5 }, svg));
      }
      const pColl = 1 - (1 - p) ** 5 - 5 * p * (1 - p) ** 4;
      txt(x0 + cols * cw / 2, 200, `collisions here: ${coll}/40   (expected ${(pColl * 100).toFixed(0)}%)`, { "text-anchor": "middle" });
      txt(x0 + cols * cw / 2, 218, "red column = 2+ features on at once", { "text-anchor": "middle", class: "muted", "font-size": 11 });
      // right: loss curves
      const gx = 360, gy = 30, gw = 260, gh = 170, X = S0 => gx + (S0 / 0.96) * gw, Y = v => gy + gh - v / 0.7 * gh;
      S("line", { x1: gx, y1: gy + gh, x2: gx + gw, y2: gy + gh, stroke: "var(--ink-3)" }, svg);
      S("line", { x1: gx, y1: gy, x2: gx, y2: gy + gh, stroke: "var(--ink-3)" }, svg);
      [0, 0.2, 0.4, 0.6].forEach(v => txt(gx - 6, Y(v) + 4, v.toFixed(1), { "text-anchor": "end", class: "muted", "font-size": 10 }));
      [0, 0.5, 0.9].forEach(v => txt(X(v), gy + gh + 14, v, { "text-anchor": "middle", class: "muted", "font-size": 10 }));
      txt(gx + gw / 2, gy + gh + 30, "sparsity S →", { "text-anchor": "middle", class: "muted", "font-size": 11 });
      txt(gx + gw / 2, 18, "signal lost (lower is better)", { "text-anchor": "middle", "font-weight": 700, "font-size": 13 });
      const line = (key, col) => S("path", { d: curve.map((c, i) => (i ? "L" : "M") + X(c.S) + "," + Y(c[key])).join(" "), fill: "none", stroke: col, "stroke-width": 2.5 }, svg);
      line("pca", "var(--blue)"); line("sup", "var(--accent)");
      txt(X(0.94), Y(curve[24].pca) - 10, "keep 2, drop 3", { fill: "var(--blue)", "font-weight": 700, "font-size": 11, "text-anchor": "end" });
      txt(X(0.94), Y(curve[24].sup) + 18, "pentagon", { fill: "var(--accent)", "font-weight": 700, "font-size": 11, "text-anchor": "end" });
      S("line", { x1: X(st.S), y1: gy, x2: X(st.S), y2: gy + gh, stroke: "var(--ink-3)", "stroke-dasharray": "3 3" }, svg);
      const sup = interp("sup"), pca = interp("pca"), b = interp("b");
      S("circle", { cx: X(st.S), cy: Y(pca), r: 5, fill: "var(--blue)" }, svg);
      S("circle", { cx: X(st.S), cy: Y(sup), r: 5, fill: "var(--accent)" }, svg);
      const win = sup < pca ? "pentagon (superposition) wins" : "keep 2, drop 3 wins";
      document.getElementById("spOut").textContent =
        `p(on) = ${p.toFixed(2)} per feature · inputs with a collision ≈ ${(pColl * 100).toFixed(0)}%\n` +
        `signal lost: keep-2 = ${(pca * 100).toFixed(0)}%, pentagon = ${(sup * 100).toFixed(0)}% (best bias ≈ ${b.toFixed(2)})  →  ${win}`;
    }
    draw();
    MI.onTheme(draw);
  })();
})();
