/* Chapter 8 widgets: toy SAE trainer, feature browser, sparsity–fidelity trade-off.
   The core toy SAE has no DOM code so it can also be tested in node. */

/* ---------------- core: data + SAE ---------------- */
function makeToySAE(opts) {
  const o = Object.assign({ nTrue: 5, m: 5, p: 0.15, lambda: 0.1, mode: "relu", k: 1, lr: 0.01, seed: 7, noise: 0.01, resample: true }, opts || {});
  let s = o.seed >>> 0;
  const rnd = () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() || 1e-9)) * Math.cos(2 * Math.PI * rnd());

  // ground-truth directions, evenly spread (5 in 2D cannot be orthogonal = superposition)
  const T = [];
  for (let j = 0; j < o.nTrue; j++) { const a = (2 * Math.PI * j) / o.nTrue + 0.3; T.push([Math.cos(a), Math.sin(a)]); }

  function sample() {
    let x0 = 0, x1 = 0; const c = [];
    for (let j = 0; j < o.nTrue; j++) {
      const v = rnd() < o.p ? rnd() : 0; c.push(v);
      x0 += v * T[j][0]; x1 += v * T[j][1];
    }
    return { x: [x0 + o.noise * gauss(), x1 + o.noise * gauss()], c };
  }

  // parameters
  const m = o.m;
  const Wd = [], We = [], be = new Array(m).fill(0), bd = [0, 0];
  // start all arrows bunched in one narrow sector, so the learning is visible
  const a0 = rnd() * 2 * Math.PI;
  for (let i = 0; i < m; i++) {
    const a = a0 + (rnd() - 0.5) * 1.0;
    Wd.push([Math.cos(a), Math.sin(a)]);
    We.push([Math.cos(a), Math.sin(a)]);   // encoder starts as decoder transpose
  }
  // Adam state
  const P = { Wd, We, be, bd };
  const M = { Wd: Wd.map(() => [0, 0]), We: We.map(() => [0, 0]), be: be.map(() => 0), bd: [0, 0] };
  const V = { Wd: Wd.map(() => [0, 0]), We: We.map(() => [0, 0]), be: be.map(() => 0), bd: [0, 0] };
  let t = 0;

  function encode(x) {
    const xb = [x[0] - bd[0], x[1] - bd[1]];
    const pre = We.map((w, i) => w[0] * xb[0] + w[1] * xb[1] + be[i]);
    let f = pre.map(v => Math.max(0, v));
    if (o.mode === "topk") {
      const idx = f.map((v, i) => i).sort((a, b) => f[b] - f[a]).slice(0, o.k);
      f = f.map((v, i) => (idx.includes(i) ? v : 0));
    }
    return { xb, pre, f };
  }
  function decode(f) {
    let y0 = bd[0], y1 = bd[1];
    f.forEach((v, i) => { if (v) { y0 += v * Wd[i][0]; y1 += v * Wd[i][1]; } });
    return [y0, y1];
  }

  // steps since each unit last fired (used for "dead" display and for resampling)
  const since = new Array(m).fill(0);
  const DEAD_AFTER = 300;

  function zeroAdam(i) { M.Wd[i] = [0, 0]; V.Wd[i] = [0, 0]; M.We[i] = [0, 0]; V.We[i] = [0, 0]; M.be[i] = 0; V.be[i] = 0; }

  // Towards Monosemanticity-style resampling: move a dead unit onto a badly rebuilt input
  function resample() {
    const alive = We.filter((w, i) => since[i] <= DEAD_AFTER);
    const avg = alive.length ? alive.reduce((a, w) => a + Math.hypot(w[0], w[1]), 0) / alive.length : 1;
    for (let i = 0; i < m; i++) {
      if (since[i] <= DEAD_AFTER) continue;
      let best = null, be2 = -1;
      for (let j = 0; j < 128; j++) {
        const { x } = sample(); const xh = decode(encode(x).f);
        const e = (x[0] - xh[0]) ** 2 + (x[1] - xh[1]) ** 2;
        if (e > be2) { be2 = e; best = x; }
      }
      const d = [best[0] - bd[0], best[1] - bd[1]], n = Math.hypot(d[0], d[1]) || 1;
      Wd[i] = [d[0] / n, d[1] / n]; We[i] = [d[0] / n * avg, d[1] / n * avg]; be[i] = 0;
      zeroAdam(i); since[i] = 0;
      resampled++;
    }
  }
  let resampled = 0;

  function step(batch) {
    const g = { Wd: Wd.map(() => [0, 0]), We: We.map(() => [0, 0]), be: be.map(() => 0), bd: [0, 0] };
    let mse = 0, l0 = 0;
    const fired = new Array(m).fill(false);
    const AUX = 1 / 8;
    for (let b = 0; b < batch; b++) {
      const { x } = sample();
      const { xb, pre, f } = encode(x);
      const xh = decode(f);
      const e = [xh[0] - x[0], xh[1] - x[1]];
      mse += e[0] * e[0] + e[1] * e[1];
      const ge = [2 * e[0], 2 * e[1]];
      g.bd[0] += ge[0]; g.bd[1] += ge[1];
      for (let i = 0; i < m; i++) {
        if (f[i] <= 0) continue;
        l0++; fired[i] = true;
        g.Wd[i][0] += ge[0] * f[i]; g.Wd[i][1] += ge[1] * f[i];
        const gp = ge[0] * Wd[i][0] + ge[1] * Wd[i][1] + (o.mode === "relu" ? o.lambda : 0);
        g.We[i][0] += gp * xb[0]; g.We[i][1] += gp * xb[1];
        g.be[i] += gp;
        g.bd[0] -= gp * We[i][0]; g.bd[1] -= gp * We[i][1];
      }
      if (o.mode === "topk") {
        // AuxK-style helper (OpenAI 2024): the best unselected unit tries to rebuild the leftover error
        let j = -1, bv = 0;
        for (let i = 0; i < m; i++) if (f[i] <= 0 && since[i] > 50) { const v = Math.max(0, pre[i]); if (v > bv) { bv = v; j = i; } }
        if (j < 0) for (let i = 0; i < m; i++) if (f[i] <= 0 && pre[i] > bv) { bv = pre[i]; j = i; }
        if (j >= 0) {
          const r = [-e[0], -e[1]];                       // what the main code missed
          const eh = [bv * Wd[j][0], bv * Wd[j][1]];
          const ga = [2 * AUX * (eh[0] - r[0]), 2 * AUX * (eh[1] - r[1])];
          g.Wd[j][0] += ga[0] * bv; g.Wd[j][1] += ga[1] * bv;
          const gp = ga[0] * Wd[j][0] + ga[1] * Wd[j][1];
          g.We[j][0] += gp * xb[0]; g.We[j][1] += gp * xb[1]; g.be[j] += gp;
        }
      }
    }
    for (let i = 0; i < m; i++) since[i] = fired[i] ? 0 : since[i] + 1;
    // Adam update
    t++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8, lr = o.lr;
    const upd = (p, gm, mm, vv, k) => {
      const gv = gm[k] / batch;
      mm[k] = b1 * mm[k] + (1 - b1) * gv;
      vv[k] = b2 * vv[k] + (1 - b2) * gv * gv;
      const mh = mm[k] / (1 - Math.pow(b1, t)), vh = vv[k] / (1 - Math.pow(b2, t));
      p[k] -= lr * mh / (Math.sqrt(vh) + eps);
    };
    for (let i = 0; i < m; i++) for (let d = 0; d < 2; d++) {
      upd(Wd[i], g.Wd[i], M.Wd[i], V.Wd[i], d);
      upd(We[i], g.We[i], M.We[i], V.We[i], d);
    }
    for (let i = 0; i < m; i++) upd(be, g.be, M.be, V.be, i);
    upd(bd, g.bd, M.bd, V.bd, 0); upd(bd, g.bd, M.bd, V.bd, 1);
    // keep decoder columns at unit length
    Wd.forEach(w => { const n = Math.hypot(w[0], w[1]) || 1; w[0] /= n; w[1] /= n; });
    if (o.resample && t % 1000 === 0 && t <= 6000) resample();
    return { mse: mse / batch, l0: l0 / batch };
  }

  function mmcs() {
    // for each true direction, best cosine with any decoder column
    let s = 0; const best = [];
    T.forEach(d => {
      let bst = -1;
      Wd.forEach(w => { bst = Math.max(bst, d[0] * w[0] + d[1] * w[1]); });
      best.push(bst); s += bst;
    });
    return { mean: s / T.length, best };
  }

  function dead() { return since.map(c => c > DEAD_AFTER); }

  return { opts: o, T, Wd, We, be, bd, sample, encode, decode, step, mmcs, dead, get t() { return t; }, get resampled() { return resampled; } };
}

if (typeof module !== "undefined") module.exports = { makeToySAE };

/* ---------------- browser widgets ---------------- */
if (typeof window !== "undefined") (function () {
  /* ========== 1. toy SAE trainer ========== */
  const root = document.getElementById("saeToy");
  if (root) {
    const svg = root.querySelector("svg.plot");
    const bars = root.querySelector("svg.bars");
    const out = root.querySelector(".readout");
    const ctl = root.querySelector(".controls");
    const state = { m: 5, lambda: 0.1, p: 0.15, mode: "relu", k: 1, running: false };
    let sae, pts, hover = null, hist = [];

    const field = (label, id, min, max, stepv, val, fmt) => {
      const l = MI.el("label", { html: `${label} <input type="range" id="${id}" min="${min}" max="${max}" step="${stepv}" value="${val}"> <output>${fmt(val)}</output>` }, ctl);
      const inp = l.querySelector("input"), o = l.querySelector("output");
      return { l, inp, o, fmt };
    };
    const fM = field("Dictionary size m", "sM", 2, 10, 1, state.m, v => (+v).toFixed(0));
    const fL = field("Sparsity penalty λ", "sL", 0, 0.6, 0.01, state.lambda, v => (+v).toFixed(2));
    const fP = field("Feature on-probability p", "sP", 0.05, 0.6, 0.01, state.p, v => (+v).toFixed(2));
    const fK = field("k (TopK)", "sK", 1, 3, 1, state.k, v => (+v).toFixed(0));
    const seg = MI.el("div", { class: "seg", html: `<button data-m="relu" class="on">ReLU + L1</button><button data-m="topk">TopK</button>` }, ctl);
    const bTrain = MI.el("button", { class: "btn primary", text: "▶ Train" }, ctl);
    const bReset = MI.el("button", { class: "btn", text: "Reset" }, ctl);

    function syncVis() { fK.l.style.display = state.mode === "topk" ? "" : "none"; fL.l.style.display = state.mode === "relu" ? "" : "none"; }
    seg.onclick = e => {
      const b = e.target.closest("button"); if (!b) return;
      state.mode = b.dataset.m;
      seg.querySelectorAll("button").forEach(x => x.classList.toggle("on", x === b));
      syncVis(); reset();
    };
    [[fM, "m"], [fL, "lambda"], [fP, "p"], [fK, "k"]].forEach(([f, key]) => {
      f.inp.oninput = () => {
        state[key] = +f.inp.value; f.o.textContent = f.fmt(f.inp.value);
        if (key === "lambda" && sae) sae.opts.lambda = state.lambda; // live change
        else reset();
      };
    });
    bTrain.onclick = () => { state.running = !state.running; bTrain.textContent = state.running ? "❚❚ Pause" : "▶ Train"; if (state.running) loop(); };
    bReset.onclick = () => reset();

    function reset() {
      sae = makeToySAE({ m: state.m, lambda: state.lambda, p: state.p, mode: state.mode, k: state.k, seed: 1 + Math.floor(Math.random() * 1e6) });
      pts = []; for (let i = 0; i < 350; i++) pts.push(sae.sample());
      hist = [];
      draw();
    }

    const S = 300, C = 150, R = 105; // plot size and scale
    const px = v => C + v[0] * R, py = v => C - v[1] * R;
    const colors = ["#e4572e", "#2563eb", "#16a34a", "#9333ea", "#0891b2", "#db2777", "#ca8a04", "#4b5563", "#65a30d", "#7c2d12"];

    function draw() {
      svg.innerHTML = "";
      // axes
      MI.svg("line", { x1: 0, y1: C, x2: S, y2: C, stroke: "var(--rule)" }, svg);
      MI.svg("line", { x1: C, y1: 0, x2: C, y2: S, stroke: "var(--rule)" }, svg);
      // data points
      pts.forEach((pt, i) => {
        MI.svg("circle", { cx: px(pt.x), cy: py(pt.x), r: i === hover ? 5 : 2.2, fill: i === hover ? "var(--accent)" : "var(--ink-3)", opacity: i === hover ? 1 : 0.45 }, svg);
      });
      // true directions
      sae.T.forEach(d => {
        MI.svg("line", { x1: C, y1: C, x2: C + d[0] * R * 1.25, y2: C - d[1] * R * 1.25, stroke: "var(--ink-3)", "stroke-width": 7, opacity: 0.18, "stroke-linecap": "round" }, svg);
      });
      // learned decoder directions
      const dead = sae.t > 30 ? sae.dead() : sae.Wd.map(() => false);
      sae.Wd.forEach((w, i) => {
        const c = colors[i % colors.length];
        MI.svg("line", { x1: C, y1: C, x2: C + w[0] * R * 1.2, y2: C - w[1] * R * 1.2, stroke: c, "stroke-width": 2.5, "stroke-dasharray": dead[i] ? "4 4" : "", opacity: dead[i] ? 0.5 : 1 }, svg);
        MI.svg("circle", { cx: C + w[0] * R * 1.2, cy: C - w[1] * R * 1.2, r: 4, fill: c }, svg);
        MI.svg("text", { x: C + w[0] * R * 1.36, y: C - w[1] * R * 1.36 + 4, "text-anchor": "middle", "font-size": 11, fill: c, "font-weight": 700 }, svg).textContent = "f" + (i + 1);
      });
      // bias point
      MI.svg("circle", { cx: px(sae.bd), cy: py(sae.bd), r: 3, fill: "none", stroke: "var(--ink)" }, svg);
      drawBars();
      const mm = sae.mmcs();
      const last = hist.length ? hist[hist.length - 1] : { mse: NaN, l0: NaN };
      const nDead = dead.filter(Boolean).length;
      out.innerHTML = `step ${sae.t} · MSE ${isNaN(last.mse) ? "–" : last.mse.toFixed(4)} · L0 (avg active features) ${isNaN(last.l0) ? "–" : last.l0.toFixed(2)} · dead ${nDead}/${sae.Wd.length}<br>` +
        `match to true directions (best cosine): ${mm.best.map(b => b.toFixed(2)).join("  ")} → mean <b>${mm.mean.toFixed(3)}</b>`;
    }

    function drawBars() {
      bars.innerHTML = "";
      const W = 300, H = 120;
      const pt = hover != null ? pts[hover] : null;
      MI.svg("text", { x: 0, y: 14, "font-size": 12, class: "muted" }, bars).textContent = pt ? "Feature activations f for the highlighted point" : "Hover over a point to see its feature activations";
      if (!pt) return;
      const { f } = sae.encode(pt.x);
      const mx = Math.max(1, ...f);
      const bw = Math.min(26, (W - 10) / f.length - 6);
      f.forEach((v, i) => {
        const h = (v / mx) * (H - 46);
        const x = 4 + i * (bw + 6);
        MI.svg("rect", { x, y: H - 18 - h, width: bw, height: Math.max(1, h), fill: v > 0 ? colors[i % colors.length] : "var(--rule)", rx: 2 }, bars);
        MI.svg("text", { x: x + bw / 2, y: H - 4, "text-anchor": "middle", "font-size": 10, class: "muted" }, bars).textContent = "f" + (i + 1);
      });
      const truth = pt.c.map((v, j) => (v > 0 ? `d${j + 1}=${v.toFixed(2)}` : null)).filter(Boolean).join(", ") || "none";
      MI.svg("text", { x: 0, y: 30, "font-size": 11, class: "muted" }, bars).textContent = "true features on: " + truth;
    }

    svg.addEventListener("mousemove", e => {
      const r = svg.getBoundingClientRect();
      const mx = ((e.clientX - r.left) / r.width) * S, my = ((e.clientY - r.top) / r.height) * S;
      let best = null, bd = 100;
      pts.forEach((p, i) => { const d = Math.hypot(px(p.x) - mx, py(p.x) - my); if (d < bd) { bd = d; best = i; } });
      hover = bd < 12 ? best : null;
      if (!state.running) draw();
    });

    function loop() {
      if (!state.running) return;
      let r;
      for (let i = 0; i < 25; i++) r = sae.step(64);
      hist.push(r);
      draw();
      if (sae.t > 20000) { state.running = false; bTrain.textContent = "▶ Train"; return; }
      requestAnimationFrame(loop);
    }
    syncVis();
    reset();
    MI.onTheme(draw);
  }

  /* ========== 2. feature browser (illustrative recreations) ========== */
  const fb = document.getElementById("featBrowser");
  if (fb) {
    const FEATS = [
      { id: "A/1/3450", name: "Arabic script", src: "Towards Monosemanticity (2023), 1-layer model",
        desc: "Fires on text in Arabic script (Arabic, Farsi, Urdu). Above an activation of about 5 it is very specific to Arabic script. No single neuron does this job.",
        text: [["The word for peace is ", 0], ["سلام", 1], [" and it is used as a ", 0], ["تحية", 0.9], [" (greeting).", 0]] },
      { id: "A/1/2937", name: "DNA sequences", src: "Towards Monosemanticity (2023)",
        desc: "Fires on long uppercase strings of the letters A, T, C, G. It was the only feature in run A/1 devoted to DNA.",
        text: [["Primer sequence: ", 0], ["ATG", 0.6], ["CGT", 0.9], ["ACC", 1], ["GAT", 1], ["TGA", 0.9], [" was used for the assay.", 0]] },
      { id: "A/1 base64", name: "base64 strings", src: "Towards Monosemanticity (2023)",
        desc: "Fires inside base64-encoded strings, for example the ID at the end of a YouTube URL. The neuron that correlates best with it (A/neurons/470, correlation 0.18) also fires on code and HTML: it is polysemantic.",
        text: [["youtube.com/watch?v=", 0.1], ["dQ", 0.8], ["w4", 1], ["w9", 0.9], ["Wg", 1], ["XcQ", 0.8], [" is the link.", 0]] },
      { id: "34M/31164353", name: "Golden Gate Bridge", src: "Scaling Monosemanticity (2024), Claude 3 Sonnet",
        desc: "Fires on mentions and descriptions of the Golden Gate Bridge, in many languages, and even on images of the bridge. Clamped to 10× its maximum, it made the model claim to be the bridge.",
        text: [["We drove north across the ", 0], ["Golden", 0.7], [" Gate", 1], [" Bridge", 1], [", and the fog rolled over the ", 0.3], ["orange towers", 0.8], [".", 0]] },
      { id: "1M/1013764", name: "Code error", src: "Scaling Monosemanticity (2024)",
        desc: "Fires on code that contains a bug or will raise an error, such as a typo in a variable name or a division by zero. Related features fire on security flaws (unsafe code, 1M/570621).",
        text: [["def mean(xs):\n  return sum(xs) / ", 0], ["len(xz)", 1], ["\n", 0]] },
      { id: "1M/847723", name: "Sycophantic praise", src: "Scaling Monosemanticity (2024)",
        desc: "Fires on over-the-top flattery. When the authors clamped it high, the model gave excessive praise to the user. Note: a feature that represents flattery does not prove the model is flattering you.",
        text: [["Your poem is ", 0], ["truly the most brilliant", 1], [" work I have ", 0.4], ["ever had the honour", 0.9], [" to read.", 0.3]] },
      { id: "1M/391411 → 1M/81163", name: "Kobe Bryant → California", src: "Scaling Monosemanticity (2024)",
        desc: "On the prompt “The capital of the state where Kobe Bryant played basketball is”, ablation showed a chain of features: Kobe Bryant, LA Lakers, Los Angeles, California, and “capital”. Together they lead to “Sacramento”.",
        text: [["The capital of the state where ", 0], ["Kobe Bryant", 1], [" played ", 0.3], ["basketball", 0.5], [" is", 0.6]] },
    ];
    const list = fb.querySelector(".fb-list"), view = fb.querySelector(".fb-view");
    function show(i) {
      const f = FEATS[i];
      list.querySelectorAll("button").forEach((b, j) => b.classList.toggle("on", j === i));
      const toks = f.text.map(([s, a]) => `<span class="tok" style="background:${a > 0 ? MI.diverge(a * 0.85) : "transparent"};${a > 0.55 ? "color:#fff" : ""}">${s.replace(/</g, "&lt;").replace(/\n/g, "<br>")}</span>`).join("");
      view.innerHTML = `<div class="fb-id">${f.id}</div><h4>${f.name}</h4><div class="fb-src">${f.src}</div><p>${f.desc}</p><div class="fb-text">${toks}</div>`;
    }
    FEATS.forEach((f, i) => { const b = MI.el("button", { html: `<b>${f.name}</b><span>${f.id}</span>` }, list); b.onclick = () => show(i); });
    show(0);
    MI.onTheme(() => { const i = [...list.children].findIndex(b => b.classList.contains("on")); show(Math.max(0, i)); });
  }

  /* ========== 3. sparsity vs reconstruction trade-off (schematic) ========== */
  const tr = document.getElementById("tradeoff");
  if (tr) {
    const svg = tr.querySelector("svg"), inp = tr.querySelector("input"), o = tr.querySelector("output"), rd = tr.querySelector(".readout");
    // schematic frontiers: x = L0 (log scale), y = fraction of variance unexplained
    const curves = [
      { name: "ReLU + L1", c: "var(--ink-3)", a: 0.75, dash: "6 4" },
      { name: "TopK / BatchTopK", c: "var(--blue)", a: 0.55 },
      { name: "JumpReLU / Gated", c: "var(--accent)", a: 0.47 },
    ];
    const X0 = 60, X1 = 560, Y0 = 250, Y1 = 30;
    const lx = l0 => X0 + ((Math.log10(l0) - 0.5) / (3 - 0.5)) * (X1 - X0);
    const yv = fvu => Y1 + (fvu / 0.45) * (Y0 - Y1);   // low error (good) at the top
    const fvuOf = (a, l0) => Math.min(0.45, a * Math.pow(l0, -0.5));
    function draw() {
      svg.innerHTML = "";
      MI.svg("line", { x1: X0, y1: Y0, x2: X1, y2: Y0, stroke: "var(--ink-3)" }, svg);
      MI.svg("line", { x1: X0, y1: Y0, x2: X0, y2: Y1, stroke: "var(--ink-3)" }, svg);
      [3, 10, 30, 100, 300, 1000].forEach(v => {
        MI.svg("line", { x1: lx(v), y1: Y0, x2: lx(v), y2: Y0 + 5, stroke: "var(--ink-3)" }, svg);
        MI.svg("text", { x: lx(v), y: Y0 + 18, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = v;
      });
      MI.svg("text", { x: (X0 + X1) / 2, y: Y0 + 36, "text-anchor": "middle", "font-size": 12 }, svg).textContent = "L0: active features per token (fewer = sparser = easier to read) →";
      MI.svg("text", { x: 18, y: (Y0 + Y1) / 2, "text-anchor": "middle", "font-size": 12, transform: `rotate(-90 18 ${(Y0 + Y1) / 2})` }, svg).textContent = "reconstruction quality ↑";
      curves.forEach((cv, ci) => {
        let d = "";
        for (let l0 = 3; l0 <= 1000; l0 *= 1.08) d += (d ? "L" : "M") + lx(l0).toFixed(1) + "," + yv(fvuOf(cv.a, l0)).toFixed(1);
        MI.svg("path", { d, fill: "none", stroke: cv.c, "stroke-width": 2.5, "stroke-dasharray": cv.dash || "" }, svg);
        MI.svg("line", { x1: X1 - 170, y1: Y0 - 70 + ci * 20, x2: X1 - 145, y2: Y0 - 70 + ci * 20, stroke: cv.c, "stroke-width": 2.5, "stroke-dasharray": cv.dash || "" }, svg);
        MI.svg("text", { x: X1 - 138, y: Y0 - 66 + ci * 20, "font-size": 12, fill: cv.c, "font-weight": 700 }, svg).textContent = cv.name;
      });
      // current λ → L0 on the ReLU curve
      const lam = +inp.value;
      const l0 = Math.pow(10, 3 - lam * 2.4);
      const fv = fvuOf(curves[0].a, l0);
      MI.svg("circle", { cx: lx(l0), cy: yv(fv), r: 7, fill: "var(--accent)", stroke: "var(--panel)", "stroke-width": 2 }, svg);
      o.textContent = lam.toFixed(2);
      let msg;
      if (lam < 0.3) msg = "Small λ: the SAE uses many features per token. The rebuild is good, but each token is a dense mix. That is hard to read — almost as bad as neurons.";
      else if (lam < 0.75) msg = "Medium λ: a few dozen features per token, and a fairly good rebuild. Most real SAEs sit in this region.";
      else msg = "Large λ: very few features per token. Each one is easy to read, but the rebuild loses a lot. Rare features die or get merged.";
      rd.textContent = `L0 ≈ ${l0.toFixed(0)} · variance unexplained ≈ ${(fv * 100).toFixed(0)}% (schematic). ${msg}`;
    }
    inp.oninput = draw;
    draw();
    MI.onTheme(draw);
  }
})();
