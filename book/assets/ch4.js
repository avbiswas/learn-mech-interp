/* Chapter 4 figures and widgets. Needs app.js (window.MI), ch4_data.js (window.CH4) and ch4_phase.js (window.CH4_PHASE).
   Every number drawn here is a real measurement; see book/scripts/ch4_experiments.py and book/scripts/ch4_phase_change.py. */
(function () {
  const S = MI.svg, D = window.CH4, P = window.CH4_PHASE;
  const fmt = (x, d = 2) => (+x).toFixed(d);
  const vis = t => t.replace(/\n/g, "⏎");

  /* ---------- shared: simple line chart ---------- */
  function lineChart(svg, o) {
    // o: {w,h,x0,y0,x1,y1, xs:[...], series:[{ys,color,dash,label}], ymax, ymin, xlabel, ylabel, xticks, yticks, bands}
    const { w, h, m = { l: 48, r: 16, t: 16, b: 34 } } = o;
    const X = v => m.l + (v - o.xmin) / (o.xmax - o.xmin) * (w - m.l - m.r);
    const Y = v => h - m.b - (Math.max(o.ymin, Math.min(o.ymax, v)) - o.ymin) / (o.ymax - o.ymin) * (h - m.t - m.b);
    (o.bands || []).forEach(b => {
      S("rect", { x: X(b.from), y: m.t, width: X(b.to) - X(b.from), height: h - m.t - m.b, fill: b.fill }, svg);
      if (b.label) S("text", { x: (X(b.from) + X(b.to)) / 2, y: m.t + 13, "text-anchor": "middle", "font-size": 11, fill: b.color || "var(--ink-3)", "font-weight": 700 }, svg).textContent = b.label;
    });
    S("line", { x1: m.l, y1: h - m.b, x2: w - m.r, y2: h - m.b, stroke: "var(--ink-3)" }, svg);
    S("line", { x1: m.l, y1: m.t, x2: m.l, y2: h - m.b, stroke: "var(--ink-3)" }, svg);
    (o.yticks || []).forEach(t => {
      S("line", { x1: m.l - 4, y1: Y(t), x2: w - m.r, y2: Y(t), stroke: "var(--rule)", "stroke-dasharray": "2 3" }, svg);
      S("text", { x: m.l - 7, y: Y(t) + 4, "text-anchor": "end", "font-size": 10.5, class: "muted" }, svg).textContent = t;
    });
    (o.xticks || []).forEach(t => S("text", { x: X(t), y: h - m.b + 15, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg).textContent = t);
    if (o.xlabel) S("text", { x: (m.l + w - m.r) / 2, y: h - 4, "text-anchor": "middle", "font-size": 11.5, class: "muted" }, svg).textContent = o.xlabel;
    if (o.ylabel) S("text", { x: 12, y: (m.t + h - m.b) / 2, "text-anchor": "middle", "font-size": 11.5, class: "muted", transform: `rotate(-90 12 ${(m.t + h - m.b) / 2})` }, svg).textContent = o.ylabel;
    o.series.forEach(s => {
      const d = s.ys.map((y, i) => `${i ? "L" : "M"}${X(o.xs[i]).toFixed(1)} ${Y(y).toFixed(1)}`).join(" ");
      S("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || "none" }, svg);
      if (s.dots) s.ys.forEach((y, i) => S("circle", { cx: X(o.xs[i]), cy: Y(y), r: 3, fill: s.color }, svg));
    });
    return { X, Y };
  }
  function legend(svg, x, y, items) {
    items.forEach((it, i) => {
      S("line", { x1: x, y1: y + i * 17, x2: x + 22, y2: y + i * 17, stroke: it.color, "stroke-width": 2.5, "stroke-dasharray": it.dash || "none" }, svg);
      S("text", { x: x + 28, y: y + i * 17 + 4, "font-size": 11.5 }, svg).textContent = it.label;
    });
  }
  function segInit(seg, fn) {
    seg.onclick = e => { const b = e.target.closest("button"); if (!b) return; [...seg.children].forEach(c => c.classList.toggle("on", c === b)); fn(b.dataset.m); };
  }

  /* ---------- Fig: loss on repeated random tokens ---------- */
  (function () {
    const svg = document.getElementById("repLossSvg"); if (!svg) return;
    function draw() {
      svg.innerHTML = "";
      const T = D.rep_T, ys = D.rep_loss.slice(1), xs = ys.map((_, i) => i + 2);   // loss for predicting positions 2..2T
      const c = lineChart(svg, { w: 680, h: 260, xmin: 1, xmax: 2 * T + 1, ymin: 0, ymax: 16, xs, series: [{ ys, color: "var(--accent)" }],
        yticks: [0, 4, 8, 12, 16], xticks: [1, 25, 51, 75, 100], xlabel: "token position", ylabel: "loss (nats)",
        bands: [{ from: 1, to: T + 0.5, fill: "var(--bg-2)", label: "first copy: tokens never seen before" }, { from: T + 0.5, to: 2 * T + 1, fill: "var(--green-soft)", label: "second copy: same tokens again", color: "var(--green)" }] });
      S("text", { x: c.X(26), y: c.Y(D.rep_loss_first) - 10, "text-anchor": "middle", "font-size": 12, "font-weight": 700 }, svg).textContent = `average ${fmt(D.rep_loss_first)} nats`;
      S("text", { x: c.X(76), y: c.Y(D.rep_loss_second) - 10, "text-anchor": "middle", "font-size": 12, "font-weight": 700, fill: "var(--green)" }, svg).textContent = `average ${fmt(D.rep_loss_second)} nats`;
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------- Widget: head score heatmaps for GPT-2 small ---------- */
  (function () {
    const svg = document.getElementById("scoreSvg"), out = document.getElementById("scoreOut"), seg = document.getElementById("scoreSeg");
    if (!svg) return;
    const info = {
      prev: { m: D.score_prev, name: "previous-token score", lo: 0, hi: 1, desc: "Average attention from each token to the token just before it." },
      induction: { m: D.score_induction, name: "induction (prefix-matching) score", lo: 0, hi: 1, desc: "In the second copy, average attention from each token to the token right AFTER its earlier copy." },
      dup: { m: D.score_dup, name: "duplicate-token score", lo: 0, hi: 1, desc: "In the second copy, average attention from each token to its own earlier copy." },
      copy: { m: D.score_copy, name: "copying score (OV circuit)", lo: -1, hi: 1, desc: "Computed from weights only: share of positive eigenvalues of W_U W_OV W_E (+1 = pure copying, −1 = pure anti-copying). Layer norm is ignored." },
    };
    let mode = "induction", sel = null;
    segInit(seg, m => { mode = m; draw(); });
    function color(v, lo) {
      if (lo < 0) return v >= 0 ? ["var(--accent)", Math.min(1, v)] : ["var(--blue)", Math.min(1, -v)];
      return ["var(--accent)", Math.min(1, v)];
    }
    function draw() {
      svg.innerHTML = "";
      const I = info[mode], cs = 30, x0 = 70, y0 = 30;
      S("text", { x: x0 + 6 * cs, y: 16, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "head →";
      for (let h = 0; h < 12; h++) S("text", { x: x0 + h * cs + cs / 2, y: y0 - 4, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg).textContent = h;
      for (let l = 0; l < 12; l++) {
        S("text", { x: x0 - 8, y: y0 + l * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 10.5, class: "muted" }, svg).textContent = "layer " + l;
        for (let h = 0; h < 12; h++) {
          const v = I.m[l][h], [c, a] = color(v, I.lo);
          const r = S("rect", { x: x0 + h * cs, y: y0 + l * cs, width: cs - 2, height: cs - 2, rx: 3, fill: "var(--bg-2)" }, svg);
          S("rect", { x: x0 + h * cs, y: y0 + l * cs, width: cs - 2, height: cs - 2, rx: 3, fill: c, opacity: 0.08 + 0.92 * a, "pointer-events": "none" }, svg);
          if (Math.abs(v) >= 0.4 && I.lo >= 0) S("text", { x: x0 + h * cs + cs / 2 - 1, y: y0 + l * cs + cs / 2 + 3, "text-anchor": "middle", "font-size": 9, fill: "#fff", "font-weight": 700, "pointer-events": "none" }, svg).textContent = fmt(v, 1).replace(/^0/, "");
          if (sel && sel[0] === l && sel[1] === h) S("rect", { x: x0 + h * cs - 1, y: y0 + l * cs - 1, width: cs, height: cs, rx: 3, fill: "none", stroke: "var(--ink)", "stroke-width": 2, "pointer-events": "none" }, svg);
          r.style.cursor = "pointer";
          r.onmouseenter = r.onclick = () => { sel = [l, h]; draw(); };
        }
      }
      const bx = x0 + 12 * cs + 24;
      S("text", { x: bx, y: y0 + 10, "font-size": 12, "font-weight": 700 }, svg).textContent = I.name;
      const words = I.desc.split(" "); let line = "", yy = y0 + 30;
      words.forEach(wd => { if ((line + wd).length > 30) { S("text", { x: bx, y: yy, "font-size": 11, class: "muted" }, svg).textContent = line; yy += 15; line = ""; } line += wd + " "; });
      S("text", { x: bx, y: yy, "font-size": 11, class: "muted" }, svg).textContent = line;
      const top = [];
      for (let l = 0; l < 12; l++) for (let h = 0; h < 12; h++) top.push([I.m[l][h], l, h]);
      top.sort((a, b) => b[0] - a[0]);
      yy += 28; S("text", { x: bx, y: yy, "font-size": 11.5, "font-weight": 700 }, svg).textContent = "Top 6 heads";
      top.slice(0, 6).forEach(([v, l, h], i) => S("text", { x: bx, y: yy + 17 + i * 16, "font-size": 11.5, "font-family": "var(--mono)" }, svg).textContent = `${l}.${h}   ${fmt(v)}`);
      if (sel) {
        const [l, h] = sel, k = `${l}.${h}`;
        const tags = [];
        if (D.induction_heads.includes(k)) tags.push("induction head");
        if (D.prev_heads.includes(k)) tags.push("previous-token head");
        out.textContent = `Head ${k}: prev-token ${fmt(D.score_prev[l][h])} · induction ${fmt(D.score_induction[l][h])} · duplicate ${fmt(D.score_dup[l][h])} · copying ${fmt(D.score_copy[l][h])} · loss rise if removed alone: +${fmt(D.ablate_each_head[l][h])} nats` + (tags.length ? `  →  ${tags.join(", ")}` : "");
      } else out.textContent = "Hover or tap a cell to see all four scores for that head.";
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------- Widget: real attention patterns on a repeated word list ---------- */
  (function () {
    const svg = document.getElementById("patSvg"), seg = document.getElementById("patSeg"), out = document.getElementById("patOut");
    if (!svg) return;
    const keys = Object.keys(D.example_attn).sort((a, b) => D.prev_heads.includes(b) - D.prev_heads.includes(a) || (+a.split(".")[0]) - (+b.split(".")[0]));
    keys.forEach((k, i) => { const b = MI.el("button", { text: k + (D.prev_heads.includes(k) ? " (prev)" : " (induction)") }, seg); b.dataset.m = k; if (i === 0) b.classList.add("on"); });
    let cur = keys[0];
    segInit(seg, m => { cur = m; draw(); });
    function draw() {
      svg.innerHTML = "";
      const toks = D.example_tokens.map((t, i) => i === 0 ? "BOS" : t.trim()), n = toks.length, cs = 25, x0 = 74, y0 = 74;
      const M = D.example_attn[cur];
      toks.forEach((t, j) => S("text", { x: x0 + j * cs + cs / 2, y: y0 - 6, "font-size": 10.5, "font-family": "var(--mono)", transform: `rotate(-55 ${x0 + j * cs + cs / 2} ${y0 - 6})` }, svg).textContent = t);
      toks.forEach((t, i) => {
        S("text", { x: x0 - 6, y: y0 + i * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 10.5, "font-family": "var(--mono)" }, svg).textContent = t;
        for (let j = 0; j < n; j++) {
          const v = j > i ? null : M[i][j];
          S("rect", { x: x0 + j * cs, y: y0 + i * cs, width: cs - 1.5, height: cs - 1.5, rx: 2, fill: v === null ? "var(--bg-2)" : (D.prev_heads.includes(cur) ? "var(--blue)" : "var(--accent)"), opacity: v === null ? 0.4 : 0.06 + 0.94 * v }, svg);
        }
      });
      S("text", { x: x0 + n * cs / 2, y: y0 + n * cs + 22, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = "key (the position attended to) →";
      S("text", { x: 14, y: y0 + n * cs / 2, "text-anchor": "middle", "font-size": 11, class: "muted", transform: `rotate(-90 14 ${y0 + n * cs / 2})` }, svg).textContent = "query (current token) →";
      const [l, h] = cur.split(".").map(Number);
      out.textContent = D.prev_heads.includes(cur)
        ? `Head ${cur} is a previous-token head: the bright cells sit on the line just left of the diagonal. Every token looks at the token before it. Previous-token score on random tokens: ${fmt(D.score_prev[l][h])}.`
        : `Head ${cur} is an induction head. In the first 8 words it mostly rests on BOS (nothing to copy yet). In the second 8 words the bright cells form a stripe 7 positions back: each word looks at the word that FOLLOWED its earlier copy. Induction score on random tokens: ${fmt(D.score_induction[l][h])}.`;
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------- Fig: ablations ---------- */
  (function () {
    const svg = document.getElementById("ablSvg"); if (!svg) return;
    const g = D.ablate_groups, nI = D.induction_heads.length, nP = D.prev_heads.length;
    const rows = [
      ["full model", g.none, "var(--green)"],
      [`remove ${nI} random other heads (avg of 5 draws)`, g.random_same_size, "var(--ink-3)"],
      [`remove the ${nP} previous-token heads`, g.prev, "var(--blue)"],
      [`remove the ${nI} induction heads`, g.induction, "var(--accent)"],
      ["reference: first copy (tokens never seen)", g.first_half_baseline, "var(--bg-2)"],
    ];
    function draw() {
      svg.innerHTML = "";
      const x0 = 300, W = 340, mx = 14;
      S("text", { x: x0, y: 16, "font-size": 11.5, class: "muted" }, svg).textContent = "loss on the second copy (nats, lower = better copying)";
      rows.forEach(([lab, v, c], i) => {
        const y = 30 + i * 38;
        S("text", { x: x0 - 10, y: y + 18, "text-anchor": "end", "font-size": 12 }, svg).textContent = lab;
        S("rect", { x: x0, y, width: Math.max(2, v / mx * W), height: 26, rx: 4, fill: c, stroke: i === 4 ? "var(--ink-3)" : "none", "stroke-dasharray": "4 3" }, svg);
        S("text", { x: x0 + Math.max(2, v / mx * W) + 6, y: y + 18, "font-size": 12, "font-weight": 700 }, svg).textContent = fmt(v);
      });
    }
    draw(); MI.onTheme(draw);
    const tb = document.getElementById("kcompTable");
    if (tb) tb.innerHTML = "<thead><tr><th>Induction head</th><th>Induction score</th><th>…after removing previous-token heads</th><th>Change</th></tr></thead><tbody>" +
      Object.entries(D.induction_after_prev_ablation).slice(0, 6).map(([k, [a, b]]) => `<tr><td><code>${k}</code></td><td>${fmt(a)}</td><td>${fmt(b)}</td><td>${fmt((b - a) / a * 100, 0)}%</td></tr>`).join("") + "</tbody>";
  })();

  /* ---------- Widget: watch induction heads form during training ---------- */
  (function () {
    const svgA = document.getElementById("phLoss"), svgB = document.getElementById("phScore");
    if (!svgA || !P) return;
    const seg = document.getElementById("phSeg"), sl = document.getElementById("phStep"), slo = document.getElementById("phStepO"), out = document.getElementById("phOut");
    let run = "two_layer";
    segInit(seg, m => { run = m; draw(); });
    sl.oninput = draw;
    function draw() {
      const R = P[run], log = R.log, xs = log.map(e => e.step), idx = +sl.value, e = log[Math.min(idx, log.length - 1)];
      slo.textContent = e.step;
      svgA.innerHTML = ""; svgB.innerHTML = "";
      const maxPrev = l => log.map(e => Math.max(...e.prev[l])), maxInd = l => log.map(e => Math.max(...e.ind[l]));
      const a = lineChart(svgA, { w: 680, h: 220, xmin: 0, xmax: 6000, ymin: 1, ymax: 4.6, xs,
        series: [{ ys: log.map(e => e.loss_fresh), color: "var(--ink-3)" }, { ys: log.map(e => e.loss_copied), color: "var(--accent)", width: 2.6 }, { ys: xs.map(() => P.chance_loss), color: "var(--ink-3)", dash: "3 4", width: 1.2 }],
        yticks: [1, 2, 3, 4], xticks: [0, 1000, 2000, 3000, 4000, 5000, 6000], ylabel: "loss (nats)" });
      legend(svgA, 470, 82, [{ color: "var(--accent)", label: "loss on copied tokens" }, { color: "var(--ink-3)", label: "loss on fresh random tokens" }, { color: "var(--ink-3)", dash: "3 4", label: `chance = ln ${P.V} = ${fmt(P.chance_loss)}` }]);
      S("line", { x1: a.X(e.step), y1: 16, x2: a.X(e.step), y2: 186, stroke: "var(--ink)", "stroke-width": 1 }, svgA);
      const ser = [{ ys: maxPrev(0), color: "var(--blue)" }, { ys: maxInd(0), color: "var(--purple)", dash: "5 4" }];
      if (run === "two_layer") ser.push({ ys: maxInd(1), color: "var(--accent)", width: 2.6 });
      const b = lineChart(svgB, { w: 680, h: 200, xmin: 0, xmax: 6000, ymin: 0, ymax: 1, xs, series: ser, yticks: [0, 0.5, 1], xticks: [0, 1000, 2000, 3000, 4000, 5000, 6000], xlabel: "training step (64 sequences of 96 tokens per step)", ylabel: "best head's score" });
      const lg = [{ color: "var(--blue)", label: "previous-token score, layer 0" }, { color: "var(--purple)", dash: "5 4", label: "induction score, layer 0" }];
      if (run === "two_layer") lg.push({ color: "var(--accent)", label: "induction score, layer 1" });
      legend(svgB, 470, 96, lg);
      S("line", { x1: b.X(e.step), y1: 16, x2: b.X(e.step), y2: 166, stroke: "var(--ink)", "stroke-width": 1 }, svgB);
      const pv = Math.max(...e.prev[0]), iv = Math.max(...e.ind[e.ind.length - 1]);
      out.textContent = `Step ${e.step}: loss on copied tokens ${fmt(e.loss_copied)}, on fresh tokens ${fmt(e.loss_fresh)} (chance ${fmt(P.chance_loss)}). Best previous-token score in layer 0: ${fmt(pv)}. Best induction score in the ${run === "two_layer" ? "second" : "only"} layer: ${fmt(iv)}.`;
    }
    sl.max = P.two_layer.log.length - 1;
    draw(); MI.onTheme(draw);
  })();

  /* ---------- Widget: MLP neurons as key-value memories ---------- */
  (function () {
    const svg = document.getElementById("kvSvg"), seg = document.getElementById("kvSeg"), box = document.getElementById("kvBox");
    if (!svg) return;
    function bars() {
      svg.innerHTML = "";
      const x0 = 50, bw = 44, H = 150, mx = 0.25, base = 180;
      S("text", { x: x0, y: 14, "font-size": 11.5, class: "muted" }, svg).textContent = "share of neurons whose value vector predicts the next token of its own trigger examples";
      [0, 0.1, 0.2].forEach(t => { S("line", { x1: x0 - 4, y1: base - t / mx * H, x2: x0 + 12 * bw, y2: base - t / mx * H, stroke: "var(--rule)", "stroke-dasharray": "2 3" }, svg); S("text", { x: x0 - 8, y: base - t / mx * H + 4, "text-anchor": "end", "font-size": 10.5, class: "muted" }, svg).textContent = (t * 100) + "%"; });
      D.kv_agreement.forEach((v, l) => {
        S("rect", { x: x0 + l * bw + 6, y: base - v / mx * H, width: bw - 12, height: v / mx * H, rx: 3, fill: l >= 8 ? "var(--accent)" : "var(--blue)", opacity: 0.85 }, svg);
        S("text", { x: x0 + l * bw + bw / 2, y: base - v / mx * H - 5, "text-anchor": "middle", "font-size": 10.5 }, svg).textContent = Math.round(v * 100) + "%";
        S("text", { x: x0 + l * bw + bw / 2, y: base + 15, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg).textContent = l;
      });
      (D.kv_baseline_by_layer || []).forEach((v, l) => {
        const yb = base - v / mx * H;
        S("line", { x1: x0 + l * bw + 2, y1: yb, x2: x0 + l * bw + bw - 2, y2: yb, stroke: "var(--ink)", "stroke-width": 2, "stroke-dasharray": "4 2" }, svg);
      });
      if (D.kv_baseline_by_layer) legend(svg, x0 + 8, 34, [{ color: "var(--ink)", dash: "4 2", label: "chance: 25 random positions instead of top triggers" }]);
      S("text", { x: x0 + 6 * bw, y: base + 32, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = "MLP layer";
    }
    const ex = D.kv_examples.filter((e, i, a) => a.findIndex(f => f.value_top[0].trim().toLowerCase() === e.value_top[0].trim().toLowerCase()) === i).slice(0, 8);
    ex.forEach((e, i) => { const b = MI.el("button", { text: e.neuron }, seg); b.dataset.m = i; if (!i) b.classList.add("on"); });
    segInit(seg, m => show(+m));
    function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
    function show(i) {
      const e = ex[i];
      box.innerHTML = `<p><b>Value</b> (what neuron ${e.neuron} writes, read through the unembedding). Top tokens it pushes up: ` +
        e.value_top.slice(0, 6).map(t => `<code>${esc(vis(t))}</code>`).join(" ") + `</p>` +
        `<p><b>Key</b> (when it fires): its top trigger contexts in WikiText, with the real next token in green. In ${e.matches_of_25} of its top 25 triggers, the next token is exactly the value's top token.</p>` +
        `<div class="table-wrap"><table class="lens-grid"><thead><tr><th>context (… up to the firing token)</th><th>next</th><th>activation</th></tr></thead><tbody>` +
        e.triggers.slice(0, 6).map(t => `<tr><td>…${esc(vis(t.ctx))}</td><td style="color:var(--green);font-weight:700">${esc(vis(t.next))}</td><td>${fmt(t.act)}</td></tr>`).join("") + `</tbody></table></div>`;
    }
    bars(); show(0); MI.onTheme(bars);
  })();

  /* ---------- Widget: direct logit attribution, attention vs MLP by layer ---------- */
  (function () {
    const svg = document.getElementById("dlaSvg"), seg = document.getElementById("dlaSeg"), out = document.getElementById("dlaOut");
    if (!svg) return;
    const opts = [{ name: "average of 16 facts", d: { attn: D.dla_mean.attn, mlp: D.dla_mean.mlp } }].concat(D.dla_prompts.slice(0, 6).map(p => ({ name: p.answer.trim(), d: p, p })));
    opts.forEach((o, i) => { const b = MI.el("button", { text: o.name }, seg); b.dataset.m = i; if (!i) b.classList.add("on"); });
    let cur = 0;
    segInit(seg, m => { cur = +m; draw(); });
    function draw() {
      svg.innerHTML = "";
      const o = opts[cur], d = o.d, x0 = 50, bw = 50, base = 150, sc = 26;
      S("line", { x1: x0, y1: base, x2: x0 + 12 * bw, y2: base, stroke: "var(--ink-3)" }, svg);
      [-2, 2, 4].forEach(t => { S("line", { x1: x0, y1: base - t * sc, x2: x0 + 12 * bw, y2: base - t * sc, stroke: "var(--rule)", "stroke-dasharray": "2 3" }, svg); S("text", { x: x0 - 6, y: base - t * sc + 4, "text-anchor": "end", "font-size": 10.5, class: "muted" }, svg).textContent = (t > 0 ? "+" : "") + t; });
      for (let l = 0; l < 12; l++) {
        [["attn", "var(--blue)", 4], ["mlp", "var(--accent)", 24]].forEach(([k, c, dx]) => {
          const v = Math.max(-2.6, Math.min(4.6, d[k][l]));
          S("rect", { x: x0 + l * bw + dx, y: v >= 0 ? base - v * sc : base, width: 19, height: Math.max(1, Math.abs(v) * sc), rx: 2, fill: c, opacity: 0.85 }, svg);
        });
        S("text", { x: x0 + l * bw + bw / 2, y: base + 84, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg).textContent = l;
      }
      S("text", { x: x0 + 6 * bw, y: base + 100, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = "layer";
      S("text", { x: 14, y: base - 20, "text-anchor": "middle", "font-size": 11, class: "muted", transform: `rotate(-90 14 ${base - 20})` }, svg).textContent = "logit contribution";
      legend(svg, x0 + 8, 18, [{ color: "var(--blue)", label: "attention block" }, { color: "var(--accent)", label: "MLP block" }]);
      const sa = d.attn.reduce((a, b) => a + b, 0), sm = d.mlp.reduce((a, b) => a + b, 0);
      out.textContent = (o.p ? `"${o.p.prompt}" → answer "${o.p.answer.trim()}" (model's top token: "${o.p.model_top.trim()}"). ` : "Averaged over 16 short fact prompts. ") +
        `Total from all attention blocks: ${fmt(sa, 1)}. Total from all MLP blocks: ${fmt(sm, 1)}. Layers 8 to 11 carry most of it.`;
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------- Fig: MLP mean-ablation ---------- */
  (function () {
    const svg = document.getElementById("mlpAblSvg"); if (!svg || !D.mlp_ablation_dlogp) return;
    function draw() {
      svg.innerHTML = "";
      const v = D.mlp_ablation_dlogp_median, x0 = 50, bw = 50, base = 40, sc = 22;
      S("line", { x1: x0, y1: base, x2: x0 + 12 * bw, y2: base, stroke: "var(--ink-3)" }, svg);
      S("text", { x: x0, y: 16, "font-size": 11.5, class: "muted" }, svg).textContent = "median change in log-probability of the right answer when one MLP is mean-ablated";
      v.forEach((x, l) => {
        const c = Math.max(-7, Math.min(2, x));
        S("rect", { x: x0 + l * bw + 8, y: c < 0 ? base : base - c * sc, width: bw - 16, height: Math.max(1.5, Math.abs(c) * sc), rx: 3, fill: c < 0 ? "var(--red)" : "var(--green)", opacity: 0.85 }, svg);
        S("text", { x: x0 + l * bw + bw / 2, y: c < 0 ? base + Math.abs(c) * sc + 14 : base - c * sc - 5, "text-anchor": "middle", "font-size": 10.5 }, svg).textContent = fmt(x, 2);
        S("text", { x: x0 + l * bw + bw / 2, y: 232, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, svg).textContent = "MLP " + l;
      });
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------- Widget: logit lens vs tuned lens on GPT-2 small ---------- */
  (function () {
    const seg = document.getElementById("lensPSeg"); if (!seg) return;
    const mseg = document.getElementById("lensSeg"), L = document.getElementById("lensL"), Lo = document.getElementById("lensLo");
    const bars = document.getElementById("lensBars"), grid = document.getElementById("lensGrid"), out = document.getElementById("lensOut");
    D.lens_prompts.forEach((p, i) => { const b = MI.el("button", { text: p.answer.trim() }, seg); b.dataset.m = i; if (!i) b.classList.add("on"); });
    let pi = 0, mode = "logit";
    segInit(seg, m => { pi = +m; draw(); });
    segInit(mseg, m => { mode = m; draw(); });
    L.oninput = draw;
    const lname = l => l === 0 ? "the embeddings" : l === 12 ? "the final output" : "after layer " + (l - 1);
    function draw() {
      const p = D.lens_prompts[pi], d = p[mode], l = +L.value;
      Lo.textContent = lname(l);
      bars.innerHTML = "";
      d[l].top.forEach(([t, pr]) => MI.el("div", { class: "lens-bar", html: `<span>${vis(t)}</span><div><div class="bar${mode === "tuned" ? " blue" : ""}" style="width:${Math.max(1, pr * 100)}%"></div></div><span>${(pr * 100).toFixed(0)}%</span>` }, bars));
      grid.innerHTML = "<tr><th>layer</th>" + d.map((_, i) => `<th>${i === 0 ? "emb" : i === 12 ? "out" : i - 1}</th>`).join("") + "</tr>" +
        "<tr><td>top-1</td>" + d.map((r, i) => `<td class="${i === l ? "hl" : ""}">${vis(r.top[0][0]).trim() || "␣"}</td>`).join("") + "</tr>" +
        `<tr><td>rank of “${p.answer.trim()}”</td>` + d.map((r, i) => `<td class="${i === l ? "hl" : ""}" style="${r.rank === 0 ? "color:var(--green);font-weight:700" : ""}">${r.rank + 1}</td>`).join("") + "</tr>";
      out.textContent = `Prompt: "${p.prompt}". Reading ${lname(l)} with the ${mode} lens: probability of "${p.answer.trim()}" is ${(d[l].p_ans * 100).toFixed(1)}%, rank ${d[l].rank + 1} of 50,257.`;
    }
    draw();
  })();

  /* ---------- Fig: logit lens vs tuned lens over a corpus ---------- */
  (function () {
    const a = document.getElementById("lensKL"), b = document.getElementById("lensAgree"); if (!a) return;
    function draw() {
      a.innerHTML = ""; b.innerHTML = "";
      const A = D.lens_aggregate, xs = [...Array(12).keys()];
      lineChart(a, { w: 330, h: 230, m: { l: 40, r: 10, t: 24, b: 38 }, xmin: 0, xmax: 11, ymin: 0, ymax: 8, xs, xticks: [0, 3, 6, 9, 11], yticks: [0, 2, 4, 6, 8], xlabel: "layer read (0 = embeddings)",
        series: [{ ys: A.logit.kl, color: "var(--accent)", dots: true }, { ys: A.tuned.kl, color: "var(--blue)", dots: true }] });
      S("text", { x: 44, y: 14, "font-size": 11.5, "font-weight": 700 }, a).textContent = "KL to final output (nats, lower = closer)";
      S("text", { x: 46, y: 36, "font-size": 10, class: "muted" }, a).textContent = `logit lens at layer 0 = ${fmt(A.logit.kl[0], 0)} (off the chart)`;
      lineChart(b, { w: 330, h: 230, m: { l: 40, r: 10, t: 24, b: 38 }, xmin: 0, xmax: 11, ymin: 0, ymax: 0.8, xs, xticks: [0, 3, 6, 9, 11], yticks: [0, 0.2, 0.4, 0.6, 0.8], xlabel: "layer read (0 = embeddings)",
        series: [{ ys: A.logit.agree, color: "var(--accent)", dots: true }, { ys: A.tuned.agree, color: "var(--blue)", dots: true }] });
      S("text", { x: 44, y: 14, "font-size": 11.5, "font-weight": 700 }, b).textContent = "top-1 agrees with the final top-1";
      legend(b, 60, 50, [{ color: "var(--accent)", label: "logit lens" }, { color: "var(--blue)", label: "tuned lens" }]);
    }
    draw(); MI.onTheme(draw);
  })();
  /* ---------- Widget: eigenvalues of a 2x2 "head" ---------- */
  (function () {
    const svg = document.getElementById("eigSvg"), out = document.getElementById("eigOut"), seg = document.getElementById("eigSeg"), ang = document.getElementById("eigAng");
    if (!svg) return;
    const mats = {
      copy: { M: [[1.4, 0.25], [0.25, 0.8]], note: "Both eigenvalues are positive: every special direction comes back the same way, stretched. Turn the input and the output never points far away. This is a copying head." },
      supp: { M: [[-1.2, 0.2], [0.2, -0.7]], note: "Both eigenvalues are negative: the special directions come back reversed. Whatever goes in, the head pushes against it. This is a suppressing (anti-copying) head." },
      mixed: { M: [[0.2, 1.1], [1.1, 0.1]], note: "One positive and one negative eigenvalue: along one direction the head copies, along the other it suppresses. The score sits near 0." },
      rot: { M: [[0.3, -1.1], [1.1, 0.3]], note: "No real eigenvectors: every input gets turned, so nothing comes back onto itself. The head maps tokens to other tokens. Only the small real part counts toward the score." },
    };
    let mode = "copy";
    segInit(seg, m => { mode = m; draw(); });
    ang.oninput = draw;
    function eig(M) {
      const a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1], tr = a + d, det = a * d - b * c, disc = tr * tr / 4 - det;
      if (disc < 0) { const re = tr / 2, im = Math.sqrt(-disc); return { complex: true, vals: [[re, im], [re, -im]] }; }
      const r = Math.sqrt(disc), l1 = tr / 2 + r, l2 = tr / 2 - r;
      const vec = l => Math.abs(b) > 1e-9 ? [b, l - a] : (Math.abs(c) > 1e-9 ? [l - d, c] : (l === a ? [1, 0] : [0, 1]));
      return { complex: false, vals: [l1, l2], vecs: [vec(l1), vec(l2)] };
    }
    function arrow(x1, y1, x2, y2, color, w) {
      S("line", { x1, y1, x2, y2, stroke: color, "stroke-width": w, "marker-end": "url(#eigHead-" + color.replace(/\W/g, "") + ")" }, svg);
    }
    function draw() {
      svg.innerHTML = "";
      const M = mats[mode].M, E = eig(M), cx = 160, cy = 150, R = 70;
      const defs = S("defs", {}, svg);
      ["var(--blue)", "var(--accent)"].forEach(col => {
        const mk = S("marker", { id: "eigHead-" + col.replace(/\W/g, ""), viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
        S("path", { d: "M0,0 L10,5 L0,10 z", fill: col }, mk);
      });
      S("circle", { cx, cy, r: R, fill: "none", stroke: "var(--rule)" }, svg);
      S("line", { x1: cx - 140, y1: cy, x2: cx + 140, y2: cy, stroke: "var(--rule)" }, svg);
      S("line", { x1: cx, y1: cy - 140, x2: cx, y2: cy + 140, stroke: "var(--rule)" }, svg);
      if (!E.complex) E.vecs.forEach(v => {
        const n = Math.hypot(v[0], v[1]), ux = v[0] / n, uy = v[1] / n;
        S("line", { x1: cx - ux * 140, y1: cy + uy * 140, x2: cx + ux * 140, y2: cy - uy * 140, stroke: "var(--ink-3)", "stroke-dasharray": "4 4" }, svg);
      });
      const t = (+ang.value) * Math.PI / 180, x = [Math.cos(t), Math.sin(t)];
      const y = [M[0][0] * x[0] + M[0][1] * x[1], M[1][0] * x[0] + M[1][1] * x[1]];
      arrow(cx, cy, cx + x[0] * R, cy - x[1] * R, "var(--blue)", 3);
      arrow(cx, cy, cx + y[0] * R, cy - y[1] * R, "var(--accent)", 3);
      const cos = (x[0] * y[0] + x[1] * y[1]) / (Math.hypot(...y) || 1);
      const g = S("g", { "font-size": 12, fill: "var(--ink-2)" }, svg);
      const line = (yy, txt, bold) => { const e = S("text", { x: 330, y: yy, "font-weight": bold ? 700 : 400, fill: "var(--ink-2)" }, g); e.textContent = txt; };
      line(40, "Matrix", true);
      line(60, `[ ${fmt(M[0][0])}  ${fmt(M[0][1])} ]`);
      line(78, `[ ${fmt(M[1][0])}  ${fmt(M[1][1])} ]`);
      line(110, "Eigenvalues", true);
      let score;
      if (E.complex) { line(130, `${fmt(E.vals[0][0])} ± ${fmt(E.vals[0][1])}i  (a turning pair)`); score = (2 * E.vals[0][0]) / (2 * Math.hypot(E.vals[0][0], E.vals[0][1])); }
      else { line(130, `λ₁ = ${fmt(E.vals[0])},  λ₂ = ${fmt(E.vals[1])}`); score = (E.vals[0] + E.vals[1]) / (Math.abs(E.vals[0]) + Math.abs(E.vals[1])); }
      line(162, "Copying score  Σλ / Σ|λ|", true);
      line(182, fmt(score));
      line(214, "This input", true);
      line(234, cos > 0.9995 ? "comes back the same way: on an eigenvector" : cos < -0.9995 ? "comes back reversed: on an eigenvector" : `comes back turned (cosine ${fmt(cos)})`);
      const k = S("g", { "font-size": 11 }, svg);
      S("text", { x: 20, y: 290, fill: "var(--blue)" }, k).textContent = "■ input";
      S("text", { x: 80, y: 290, fill: "var(--accent)" }, k).textContent = "■ output";
      S("text", { x: 150, y: 290, fill: "var(--ink-3)" }, k).textContent = "- - eigenvector directions";
      out.textContent = mats[mode].note;
    }
    draw(); MI.onTheme(draw);
  })();
})();
