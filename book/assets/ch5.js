/* Chapter 5 widgets and procedural figures. Depends on app.js (window.MI) and ioi_patching.js. */
(function () {
  const S = MI.svg;
  const D = window.IOI_PATCHING || [];

  /* ---------------- Figure 5.2: three runs ---------------- */
  (function () {
    const rows = [
      { g: "rowClean", y: 40, text: "…store, John gave a drink to", col: "var(--green)", out: "→ Mary", outCol: "var(--green)" },
      { g: "rowCorr", y: 130, text: "…store, Mary gave a drink to", col: "var(--red)", out: "→ John", outCol: "var(--red)" },
      { g: "rowPatch", y: 220, text: "…store, Mary gave a drink to", col: "var(--red)", out: "→ Mary?", outCol: "var(--accent)" },
    ];
    rows.forEach((r, ri) => {
      const g = document.getElementById(r.g);
      if (!g) return;
      S("text", { x: 95, y: r.y + 4, "font-family": "var(--mono)", "font-size": 11.5 }, g).textContent = r.text;
      for (let i = 0; i < 5; i++) {
        const x = 301 + i * 40;
        if (i < 4) S("line", { x1: x + 11, y1: r.y, x2: x + 29, y2: r.y, stroke: "var(--ink-3)" }, g);
        const isPatch = ri === 2 && i === 2;
        S("circle", { cx: x, cy: r.y, r: 11, fill: isPatch ? "var(--green-soft)" : "var(--panel)", stroke: isPatch ? "var(--green)" : r.col, "stroke-width": isPatch ? 3 : 1.6 }, g);
        if (ri === 0) S("text", { x, y: r.y - 17, "text-anchor": "middle", "font-size": 10, class: "muted" }, g).textContent = "L" + Math.min(11, i * 3);
      }
      S("text", { x: 500, y: r.y + 5, "font-weight": 700, fill: r.outCol, "font-size": 13 }, g).textContent = r.out;
    });
  })();

  /* ---------------- Figure 5.4: ROME schematic ---------------- */
  (function () {
    const svg = document.getElementById("romeFig");
    if (!svg) return;
    const toks = ["The*", "Space*", "Need*", "le*", "is", "in", "downtown"];
    const L = 48, x0 = 90, y0 = 20, cw = (700 - x0) / L, ch = 28;
    function val(t, l) {
      if (t === 3) return Math.max(0, 0.95 * Math.exp(-Math.pow((l - 10) / 9, 2) / 1) + (l < 8 ? 0.5 : 0));
      if (t === 6) return 0.9 / (1 + Math.exp(-(l - 24) / 3));
      if (t < 3) return l < 3 ? 0.25 : 0.04;
      return 0.03;
    }
    function draw() {
      svg.innerHTML = "";
      toks.forEach((tk, t) => {
        S("text", { x: x0 - 8, y: y0 + t * ch + 18, "text-anchor": "end", "font-size": 12, "font-family": "var(--mono)" }, svg).textContent = tk;
        for (let l = 0; l < L; l++) S("rect", { x: x0 + l * cw, y: y0 + t * ch, width: cw + 0.3, height: ch - 2, fill: MI.diverge(Math.min(1, val(t, l))) }, svg);
      });
      [0, 10, 20, 30, 40, 47].forEach(l => S("text", { x: x0 + l * cw + cw / 2, y: y0 + toks.length * ch + 14, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = l);
      S("text", { x: 395, y: y0 + toks.length * ch + 32, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "layer restored (GPT-2 XL has 48)";
      S("text", { x: x0 + 12 * cw, y: y0 + 3 * ch - 6, "font-size": 12, "font-weight": 700, fill: "var(--accent)" }, svg).textContent = "early site (mid-layer MLPs)";
      S("text", { x: x0 + 30 * cw, y: y0 + 6 * ch - 6, "font-size": 12, "font-weight": 700, fill: "var(--accent)" }, svg).textContent = "late site";
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------------- Figure 5.6: causal scrubbing rows ---------------- */
  (function () {
    const g = document.getElementById("scrubRows");
    if (!g) return;
    const rows = [
      ["…the cat sat", "var(--blue)", "prev = cat"],
      ["…a cat ran", "var(--blue)", "prev = cat"],
      ["…the dog sat", "var(--green)", "prev = dog"],
      ["…my dog ran", "var(--green)", "prev = dog"],
    ];
    rows.forEach((r, i) => {
      const y = 50 + i * 40;
      S("rect", { x: 20, y: y, width: 200, height: 30, rx: 6, fill: "var(--panel)", stroke: r[1], "stroke-width": 2 }, g);
      S("text", { x: 32, y: y + 20, "font-family": "var(--mono)", "font-size": 12 }, g).textContent = r[0];
      S("text", { x: 236, y: y + 20, "font-size": 12, fill: r[1], "font-weight": 700 }, g).textContent = r[2];
    });
    [[0, 1, "var(--blue)"], [2, 3, "var(--green)"]].forEach(([a, b, c]) => {
      const ya = 65 + a * 40, yb = 65 + b * 40;
      S("path", { d: `M 330 ${ya} C 380 ${ya}, 380 ${yb}, 330 ${yb}`, stroke: c, fill: "none", "stroke-width": 2, "stroke-dasharray": "4 3" }, g);
      S("text", { x: 388, y: (ya + yb) / 2 + 4, "font-size": 12, fill: c }, g).textContent = "swap head X";
    });
  })();

  /* ---------------- Figure 5.7: DAS points ---------------- */
  (function () {
    const L = document.getElementById("dasLeft"), R = document.getElementById("dasRight");
    if (!L) return;
    const r = MI.rng(7);
    for (let i = 0; i < 26; i++) {
      const s = r() * 2 - 1, n = MI.gauss(r) * 0.18;
      const big = s > 0;
      // along a 35-degree direction
      const c = Math.cos(35 * Math.PI / 180), sn = Math.sin(35 * Math.PI / 180);
      const u = s * 1.0, v = n;
      const x = u * c - v * sn, y = u * sn + v * c;
      S("circle", { cx: 175 + x * 90, cy: 125 - y * 90, r: 4.5, fill: big ? "var(--accent)" : "var(--blue)", opacity: 0.85 }, L);
      const ang = -35 * Math.PI / 180;
      const X = u * 100, Y = v * 100;
      S("circle", { cx: 560 + X * Math.cos(ang) - Y * Math.sin(ang), cy: 130 + X * Math.sin(ang) + Y * Math.cos(ang), r: 4.5, fill: big ? "var(--accent)" : "var(--blue)", opacity: 0.85 }, R);
    }
  })();

  /* ---------------- Figure 5.8: DLA waterfall (schematic) ---------------- */
  (function () {
    const svg = document.getElementById("dlaFig");
    if (!svg) return;
    const items = [["9.9", 0.8], ["9.6", 0.6], ["10.0", 0.5], ["backup", 0.5], ["7.3 (S-inh.)", 0.02], ["MLPs etc.", 0.2], ["10.7", -0.4], ["11.10", -0.2]];
    function draw() {
      svg.innerHTML = "";
      const x0 = 50, bw = 62, gap = 12, base = 225, sc = 60;
      S("line", { x1: x0 - 10, y1: base, x2: 700, y2: base, stroke: "var(--ink-3)" }, svg);
      let run = 0;
      items.forEach((it, i) => {
        const x = x0 + i * (bw + gap), v = it[1];
        const y1 = base - run * sc, y2 = base - (run + v) * sc;
        S("rect", { x, y: Math.min(y1, y2), width: bw, height: Math.max(2, Math.abs(y1 - y2)), rx: 3, fill: v >= 0 ? "var(--accent)" : "var(--blue)", opacity: 0.85 }, svg);
        S("text", { x: x + bw / 2, y: base + 16, "text-anchor": "middle", "font-size": 11 }, svg).textContent = it[0];
        S("text", { x: x + bw / 2, y: Math.min(y1, y2) - 5, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = (v > 0 ? "+" : "") + v.toFixed(2);
        run += v;
      });
      const x = x0 + items.length * (bw + gap);
      S("rect", { x, y: base - run * sc, width: bw, height: run * sc, rx: 3, fill: "var(--ink-2)" }, svg);
      S("text", { x: x + bw / 2, y: base + 16, "text-anchor": "middle", "font-size": 11, "font-weight": 700 }, svg).textContent = "total";
      S("text", { x: x + bw / 2, y: base - run * sc - 5, "text-anchor": "middle", "font-size": 11, "font-weight": 700 }, svg).textContent = run.toFixed(2);
      S("text", { x: 40, y: 18, "font-size": 12, class: "muted" }, svg).textContent = "direct contribution to logit(Mary) − logit(John)   (illustrative values)";
    }
    draw(); MI.onTheme(draw);
  })();

  /* ---------------- Ablation widget ---------------- */
  (function () {
    const svg = document.getElementById("ablSvg");
    if (!svg) return;
    const r = MI.rng(3);
    const data = Array.from({ length: 400 }, () => 5 + 0.25 * MI.gauss(r));
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    const LD = a => 3 + 0.6 * (a - 5) - 3 * Math.max(0, Math.abs(a - 5) - 1.5);
    let cur = 5 + 0.25 * MI.gauss(r), type = "none", resampled = data[0];
    const bars = document.getElementById("ablBars"), out = document.getElementById("ablOut");
    const xs = a => 40 + a / 7 * 560;
    function draw() {
      svg.innerHTML = "";
      // histogram
      const bins = new Array(70).fill(0);
      data.forEach(a => { const b = Math.floor(a * 10); if (b >= 0 && b < 70) bins[b]++; });
      const mx = Math.max(...bins);
      bins.forEach((c, i) => c && S("rect", { x: xs(i / 10), y: 150 - c / mx * 100, width: 560 / 70 - 0.5, height: c / mx * 100, fill: "var(--ink-3)", opacity: 0.45 }, svg));
      S("line", { x1: 40, y1: 150, x2: 600, y2: 150, stroke: "var(--ink-3)" }, svg);
      for (let t = 0; t <= 7; t++) S("text", { x: xs(t), y: 168, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = t;
      S("text", { x: 320, y: 188, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "neuron activation (grey = values seen on 400 normal inputs)";
      S("rect", { x: xs(3.5), y: 40, width: xs(6.5) - xs(3.5), height: 110, fill: "var(--green)", opacity: 0.07 }, svg);
      S("text", { x: xs(6.5) + 6, y: 52, "font-size": 11, fill: "var(--green)" }, svg).textContent = "expected range";
      const mark = (a, col, lab, dy) => {
        S("line", { x1: xs(a), y1: 32 + dy, x2: xs(a), y2: 150, stroke: col, "stroke-width": 2.5 }, svg);
        S("text", { x: xs(a) + (a < 1 ? 6 : -6), y: 26 + dy, "text-anchor": a < 1 ? "start" : "end", "font-size": 12, "font-weight": 700, fill: col }, svg).textContent = lab;
      };
      mark(cur, "var(--ink)", "this input", 0);
      const rep = repVal();
      if (type !== "none") mark(rep, "var(--accent)", "replaced with", type === "zero" ? 0 : 20);
      // bars
      const clean = LD(cur), abl = LD(type === "none" ? cur : rep), dmg = Math.abs(clean - abl);
      const rowsHtml = [["Output, normal", clean, "var(--green)"], ["Output, ablated", abl, abl < 0 ? "var(--blue)" : "var(--accent)"], ["Damage reported", dmg, "var(--red)"]].map(([n, v, c]) => {
        const w = Math.min(100, Math.abs(v) / 11 * 100);
        return `<div class="barrow"><span>${n}</span><div class="track"><div class="fill" style="left:0;width:${w}%;background:${c}"></div></div><span class="val">${v.toFixed(2)}</span></div>`;
      }).join("");
      bars.innerHTML = rowsHtml;
      const msg = {
        none: "No ablation. The neuron works normally.",
        zero: "Zero ablation puts the neuron far outside its normal range. The rest of the model breaks, so the damage looks huge. But most of what we removed was a constant, not information about the input.",
        mean: "Mean ablation keeps the constant part and removes only the input-specific part. The damage is small. This is the honest answer: the neuron carries little task information.",
        resample: "Resample ablation uses the neuron's value on another random input. The value is realistic, so the damage is small. It is noisy: press 'New input' a few times.",
      };
      out.textContent = msg[type];
    }
    function repVal() { return type === "zero" ? 0 : type === "mean" ? mean : resampled; }
    document.querySelectorAll("#ablType button").forEach(b => b.onclick = () => {
      document.querySelectorAll("#ablType button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); type = b.dataset.t;
      if (type === "resample") resampled = data[Math.floor(r() * data.length)];
      draw();
    });
    document.getElementById("ablNew").onclick = () => { cur = 5 + 0.25 * MI.gauss(r); resampled = data[Math.floor(r() * data.length)]; draw(); };
    draw(); MI.onTheme(draw);
  })();

  /* ---------------- Real patching heatmap ---------------- */
  const ROLES = {
    "9.9": ["Name mover", "Attends from the final token to the names and copies the indirect object (Mary) to the output."],
    "9.6": ["Name mover", "Copies the indirect object name to the output."],
    "10.0": ["Name mover", "Copies the indirect object name to the output."],
    "10.7": ["Negative name mover", "Attends to names and pushes DOWN the copied name. Later shown to do 'copy suppression'."],
    "11.10": ["Negative name mover", "Pushes against the predicted name; hedges the answer."],
    "7.3": ["S-inhibition", "Reads the 'duplicate name' signal at S2 and tells name movers NOT to attend to the repeated name (John)."],
    "7.9": ["S-inhibition", "Moves the 'John is repeated' signal to the final token."],
    "8.6": ["S-inhibition", "Moves the 'John is repeated' signal to the final token."],
    "8.10": ["S-inhibition", "Moves the 'John is repeated' signal to the final token."],
    "10.10": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated (self-repair)."],
    "10.6": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "10.2": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "10.1": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "11.2": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "9.7": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "9.0": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "11.9": ["Backup name mover", "Normally quiet. Takes over when name movers are ablated."],
    "5.5": ["Induction head", "At S2, attends to the token after S1 and helps flag that 'John' appeared before."],
    "6.9": ["Induction head", "Helps detect the repeated name."],
    "5.8": ["Induction head", "Helps detect the repeated name."],
    "5.9": ["Induction head", "Helps detect the repeated name."],
    "0.1": ["Duplicate token head", "At S2, attends to the earlier copy of the same token (S1)."],
    "0.10": ["Duplicate token head", "At S2, attends to the earlier copy of the same token."],
    "3.0": ["Duplicate token head", "At S2, attends to S1: signals 'this name is a duplicate'."],
    "2.2": ["Previous token head", "Copies information about the previous token; feeds induction heads."],
    "4.11": ["Previous token head", "Copies information about the previous token; feeds induction heads."],
  };
  const ROLE_COL = { "Name mover": "var(--accent)", "Negative name mover": "var(--blue)", "S-inhibition": "var(--purple)", "Backup name mover": "var(--yellow)", "Induction head": "var(--green)", "Duplicate token head": "var(--green)", "Previous token head": "var(--green)" };

  let corrIdx = 0, comp = "resid_pre", sharedScale = false, headKey = "heads_all_pos", hoverPos = -1;

  function legend(el, scale, note) {
    el.innerHTML = `<span>${(-scale).toFixed(1)}</span><span class="grad" style="background:linear-gradient(90deg,${MI.diverge(-1)},${MI.diverge(0)},${MI.diverge(1)})"></span><span>+${scale.toFixed(1)}</span><span>· 0 = corrupted baseline, 1 = fully clean${note ? " · " + note : ""}</span>`;
  }

  const absMax = m => Math.max(...m.flat().map(Math.abs));

  function sentence(el, toks, other, hl) {
    el.innerHTML = toks.map((t, i) => {
      const cls = [other && other[i] !== t ? "diff" : "", i === hl ? "hl" : ""].join(" ").trim();
      return `<span class="${cls}">${t.replace(/ /g, "&nbsp;")}</span>`;
    }).join("");
  }

  function drawPatch() {
    const svg = document.getElementById("pwSvg");
    if (!svg || !D.length) return;
    const r = D[corrIdx], M = r[comp];
    const T = r.tokens.length, x0 = 70, y0 = 34, cw = (890 - x0) / T, ch = 27;
    svg.setAttribute("viewBox", `0 0 900 ${y0 + 12 * ch + 30}`);
    svg.innerHTML = "";
    // Colour range: fit this heatmap, or share one range across both corruptions
    // (same component) so switching corruption doesn't silently rescale the colours.
    const mx = Math.max(1, ...(sharedScale ? D.map(d => absMax(d[comp])) : [absMax(M)]));
    legend(document.getElementById("pwLegend"), mx, sharedScale ? "same range for both corruptions" : "range fitted to this heatmap");
    r.tokens.forEach((t, p) => {
      const diff = r.corr_tokens[p] !== t;
      S("text", { x: x0 + p * cw + cw / 2, y: y0 - 10, "text-anchor": "middle", "font-size": 11.5, "font-family": "var(--mono)", "font-weight": diff ? 700 : 400, fill: diff ? "var(--yellow)" : "var(--ink)" }, svg).textContent = t.trim() || "␣";
    });
    const tags = { 1: "IO", 3: "S1", 9: "S2", 13: "END" };
    for (let L = 0; L < 12; L++) {
      S("text", { x: x0 - 8, y: y0 + L * ch + ch / 2 + 4, "text-anchor": "end", "font-size": 11.5, class: "muted" }, svg).textContent = "L" + L;
      for (let p = 0; p < T; p++) {
        const v = M[L][p];
        const c = S("rect", { class: "hm-cell", x: x0 + p * cw + 1, y: y0 + L * ch + 1, width: cw - 2, height: ch - 2, rx: 3, fill: MI.diverge(v / mx) }, svg);
        c.onmouseenter = () => {
          document.getElementById("pwOut").innerHTML = `Layer ${L}, position ${p} ("${r.tokens[p]}") · ${comp} · normalized score = <b>${v.toFixed(3)}</b>` +
            `<br>Patched logit diff ≈ ${(r.corrupted_logit_diff + v * (r.clean_logit_diff - r.corrupted_logit_diff)).toFixed(2)} (corrupted ${r.corrupted_logit_diff.toFixed(2)}, clean ${r.clean_logit_diff.toFixed(2)})`;
          sentence(document.getElementById("pwClean"), r.tokens, null, p);
        };
        if (Math.abs(v) >= 0.25) S("text", { x: x0 + p * cw + cw / 2, y: y0 + L * ch + ch / 2 + 4, "text-anchor": "middle", "font-size": 10, fill: Math.abs(v / mx) > 0.55 ? "#fff" : "var(--ink)", "pointer-events": "none" }, svg).textContent = v.toFixed(2);
      }
    }
    Object.entries(tags).forEach(([p, t]) => S("text", { x: x0 + p * cw + cw / 2, y: y0 + 12 * ch + 16, "text-anchor": "middle", "font-size": 10.5, "font-weight": 700, class: "muted" }, svg).textContent = t);
    sentence(document.getElementById("pwClean"), r.tokens, null, -1);
    sentence(document.getElementById("pwCorrS"), r.corr_tokens, r.tokens, -1);
    const ex = {
      "0resid_pre": "S2 swap, residual stream: the signal sits at S2 (the second ' John') from layer 0 to about layer 7. Between layers 7 and 9 it moves to the final token. Attention heads (the S-inhibition heads) carry it there.",
      "0attn_out": "S2 swap, attention outputs: the attention blocks that matter write at the END position in layers 7–9 — the S-inhibition heads moving the 'John is repeated' signal, and name movers in layer 9.",
      "0mlp_out": "S2 swap, MLP outputs: MLP 0 at S2 matters most (score near 1): GPT-2's first MLP acts like an extra embedding for the token. Other MLPs matter much less.",
      "1resid_pre": "Replace-all-names corruption ('When Alice and Bob went to the store, Bob gave a drink to'): IO, S1 and S2 all differ, but the repeat pattern is the same. Early on the IO column carries the answer's identity. S1 is negative: restoring only the first 'John' confuses the model. Late layers focus on END.",
      "1attn_out": "Replace-all-names corruption, attention outputs: the big contribution is at END in layer 9 (name movers 9.6 and 9.9 copying 'Mary'), with a smaller one in layer 7. Layer 10 sums to near zero at END, because name mover 10.0 and negative head 10.7 cancel.",
      "1mlp_out": "Replace-all-names corruption, MLP outputs: the MLP at layer 0 on the name tokens matters a lot. Early MLPs help build each name's representation (GPT-2's first MLP acts like an extended embedding).",
    };
    document.getElementById("pwExplain").textContent = ex[corrIdx + comp] || "";
  }

  function drawHeads() {
    const svg = document.getElementById("hwSvg");
    if (!svg || !D.length) return;
    const r = D[corrIdx], M = r[headKey];
    const x0 = 46, y0 = 28, c = 34;
    svg.innerHTML = "";
    const mx = Math.max(0.3, ...(sharedScale ? D.map(d => absMax(d[headKey])) : [absMax(M)]));
    for (let h = 0; h < 12; h++) S("text", { x: x0 + h * c + c / 2, y: y0 - 8, "text-anchor": "middle", "font-size": 11, class: "muted" }, svg).textContent = h;
    for (let L = 0; L < 12; L++) {
      S("text", { x: x0 - 8, y: y0 + L * c + c / 2 + 4, "text-anchor": "end", "font-size": 11, class: "muted" }, svg).textContent = "L" + L;
      for (let h = 0; h < 12; h++) {
        const key = L + "." + h, v = M[L][h], role = ROLES[key];
        const rect = S("rect", { class: "hm-cell", x: x0 + h * c + 1, y: y0 + L * c + 1, width: c - 2, height: c - 2, rx: 4, fill: MI.diverge(v / mx), stroke: role ? "var(--ink)" : "none", "stroke-width": role ? 1.4 : 0 }, svg);
        rect.onmouseenter = () => showHead(key, v);
      }
    }
    S("text", { x: x0, y: y0 + 12 * c + 20, "font-size": 11.5, class: "muted" }, svg).textContent = `colour scale ±${mx.toFixed(2)} (${sharedScale ? "shared" : "fitted"}) · outlined = named IOI role`;
    if (lastHead) { const [lL, lH] = lastHead.split(".").map(Number); showHead(lastHead, M[lL][lH]); }
  }

  let lastHead = null;
  function showHead(key, v) {
    lastHead = key;
    const role = ROLES[key];
    const box = document.getElementById("hwRole");
    box.innerHTML = `<b>Head ${key}</b> · score <b>${v.toFixed(3)}</b><br>` +
      (role ? `<span style="color:${ROLE_COL[role[0]]};font-weight:700">${role[0]}</span><br>${role[1]}` : `<span class="muted">No named role in the IOI circuit.</span>`);
    const r = D[corrIdx], pat = r.attn_patterns_final[key];
    const ps = document.getElementById("hwPat"), lab = document.getElementById("hwPatLabel");
    ps.innerHTML = "";
    if (!pat) { lab.textContent = ""; return; }
    lab.textContent = `Attention of ${key} from the final token (clean run)`;
    const bw = 370 / r.tokens.length, ox = 16;
    pat.forEach((a, i) => {
      S("rect", { x: ox + i * bw + 2, y: 140 - a * 130, width: bw - 4, height: Math.max(1, a * 130), rx: 2, fill: "var(--accent)", opacity: 0.85 }, ps);
      S("text", { x: ox + i * bw + bw / 2, y: 154, "text-anchor": "end", "font-size": 10, transform: `rotate(-45 ${ox + i * bw + bw / 2} 154)`, "font-family": "var(--mono)" }, ps).textContent = r.tokens[i].trim();
      if (a > 0.08) S("text", { x: ox + i * bw + bw / 2, y: 135 - a * 130, "text-anchor": "middle", "font-size": 9.5, class: "muted" }, ps).textContent = a.toFixed(2);
    });
  }

  function seg(id, fn) {
    const el = document.getElementById(id);
    if (!el) return;
    el.querySelectorAll("button").forEach(b => b.onclick = () => {
      el.querySelectorAll("button").forEach(x => x.classList.remove("on"));
      b.classList.add("on"); fn(b.dataset.v);
    });
  }
  function syncCorr(v) {
    corrIdx = +v;
    ["pwCorr", "hwCorr"].forEach(id => document.querySelectorAll(`#${id} button`).forEach(b => b.classList.toggle("on", b.dataset.v === v)));
    drawPatch(); drawHeads();
  }
  seg("pwCorr", syncCorr);
  seg("hwCorr", syncCorr);
  seg("pwComp", v => { comp = v; drawPatch(); });
  function syncScale(v) {
    sharedScale = v === "shared";
    ["pwScale", "hwScale"].forEach(id => document.querySelectorAll(`#${id} button`).forEach(b => b.classList.toggle("on", b.dataset.v === v)));
    drawPatch(); drawHeads();
  }
  seg("pwScale", syncScale);
  seg("hwScale", syncScale);
  seg("hwPos", v => { headKey = v; drawHeads(); });
  drawPatch(); drawHeads();
  if (D.length) showHead("9.9", D[0].heads_all_pos[9][9]);
  MI.onTheme(() => { drawPatch(); drawHeads(); });

  /* ---------------- ACDC toy ---------------- */
  (function () {
    const svg = document.getElementById("acdcSvg");
    if (!svg) return;
    const N = {
      in: [320, 300, "input"], a1: [140, 220, "A1"], a2: [320, 220, "A2"], a3: [500, 220, "A3"],
      b1: [140, 120, "B1"], b2: [320, 120, "B2"], b3: [500, 120, "B3"], out: [320, 30, "output"],
    };
    const E0 = [
      ["b1", "out", 0.30], ["b2", "out", 0.04], ["b3", "out", 0.22], ["a1", "out", 0.02], ["a2", "out", 0.12], ["a3", "out", 0.01],
      ["a1", "b1", 0.25], ["a2", "b1", 0.03], ["a3", "b1", 0.06], ["a1", "b2", 0.05], ["a2", "b2", 0.02], ["a3", "b2", 0.04],
      ["a1", "b3", 0.01], ["a2", "b3", 0.07], ["a3", "b3", 0.18],
      ["in", "a1", 0.4], ["in", "a2", 0.3], ["in", "a3", 0.2],
    ];
    let E, idx, kl, done, timer;
    const tauEl = document.getElementById("acdcTau"), tauO = document.getElementById("acdcTauO"), out = document.getElementById("acdcOut");
    function reset() {
      clearInterval(timer);
      E = E0.map(e => ({ from: e[0], to: e[1], eff: e[2], state: "alive" }));
      idx = 0; kl = 0; done = false; draw("Ready. ACDC will visit edges from the output backwards.");
    }
    function alive(e) { return e.state !== "removed"; }
    function nodeUsed(n) { return n === "out" || E.some(e => e.from === n && alive(e)); }
    function step() {
      if (done) return;
      const tau = +tauEl.value;
      while (idx < E.length && E[idx].state === "removed") idx++;
      if (idx >= E.length) { done = true; clearInterval(timer); draw("Done. The edges that remain are the circuit."); return; }
      const e = E[idx];
      let msg;
      if (!nodeUsed(e.to)) { e.state = "removed"; msg = `${N[e.from][2]} → ${N[e.to][2]}: ${N[e.to][2]} no longer reaches the output, so this edge is removed for free.`; }
      else if (e.eff < tau) { e.state = "removed"; kl += e.eff; msg = `${N[e.from][2]} → ${N[e.to][2]}: corrupting it raises KL by ${e.eff.toFixed(2)} < τ. Removed.`; }
      else { e.state = "kept"; msg = `${N[e.from][2]} → ${N[e.to][2]}: corrupting it raises KL by ${e.eff.toFixed(2)} ≥ τ. Kept.`; }
      idx++;
      draw(msg, e);
    }
    function draw(msg, cur) {
      svg.innerHTML = "";
      E.forEach(e => {
        const [x1, y1] = N[e.from], [x2, y2] = N[e.to];
        const col = e === cur ? "var(--accent)" : e.state === "kept" ? "var(--green)" : "var(--ink-3)";
        S("line", { x1, y1: y1 - 16, x2, y2: y2 + 16, stroke: col, "stroke-width": e.state === "removed" ? 1 : 1.5 + e.eff * 10, opacity: e.state === "removed" ? 0.12 : 0.9, "stroke-dasharray": e.state === "removed" ? "3 4" : "" }, svg);
      });
      Object.entries(N).forEach(([k, [x, y, lab]]) => {
        const used = nodeUsed(k) || k === "in";
        S("rect", { x: x - 40, y: y - 16, width: 80, height: 32, rx: 16, fill: used ? "var(--panel)" : "var(--bg-2)", stroke: k === "out" ? "var(--accent)" : "var(--ink-3)", opacity: used ? 1 : 0.4 }, svg);
        S("text", { x, y: y + 5, "text-anchor": "middle", "font-size": 13, opacity: used ? 1 : 0.4 }, svg).textContent = lab;
      });
      const kept = E.filter(alive).length;
      out.innerHTML = `${msg}<br>Edges left: <b>${kept}</b> / ${E.length} · total KL added by removals ≈ <b>${kl.toFixed(2)}</b>`;
    }
    tauEl.oninput = () => { tauO.textContent = (+tauEl.value).toFixed(2); reset(); };
    document.getElementById("acdcStep").onclick = step;
    document.getElementById("acdcRun").onclick = () => { clearInterval(timer); timer = setInterval(step, 450); };
    document.getElementById("acdcReset").onclick = reset;
    reset(); MI.onTheme(() => draw("Theme changed."));
  })();

  /* ---------------- Tangent / attribution patching ---------------- */
  (function () {
    const svg = document.getElementById("tanSvg");
    if (!svg) return;
    const kEl = document.getElementById("tanK"), cEl = document.getElementById("tanC"), nEl = document.getElementById("tanN");
    const sig = z => 1 / (1 + Math.exp(-z));
    function f(a, k, c) {
      if (k < 0.01) return a;
      const lo = sig(-k * c), hi = sig(k * (1 - c));
      return (sig(k * (a - c)) - lo) / (hi - lo);
    }
    function df(a, k, c) { const h = 1e-4; return (f(a + h, k, c) - f(a - h, k, c)) / (2 * h); }
    function draw() {
      const k = +kEl.value, c = +cEl.value, n = +nEl.value;
      document.getElementById("tanKO").textContent = k; document.getElementById("tanCO").textContent = c; document.getElementById("tanNO").textContent = n;
      svg.innerHTML = "";
      const X = a => 60 + a * 520, Y = v => 240 - v * 190;
      S("line", { x1: X(0), y1: Y(0), x2: X(1.02), y2: Y(0), stroke: "var(--ink-3)" }, svg);
      S("line", { x1: X(0), y1: Y(-0.05), x2: X(0), y2: Y(1.1), stroke: "var(--ink-3)" }, svg);
      S("text", { x: X(0), y: 262, "text-anchor": "middle", "font-size": 11.5, class: "muted" }, svg).textContent = "α = 0 (corrupted)";
      S("text", { x: X(1), y: 262, "text-anchor": "middle", "font-size": 11.5, class: "muted" }, svg).textContent = "α = 1 (clean)";
      S("text", { x: X(0) - 8, y: Y(1) + 4, "text-anchor": "end", "font-size": 11, class: "muted" }, svg).textContent = "f(1)";
      S("text", { x: X(0) - 8, y: Y(0) + 4, "text-anchor": "end", "font-size": 11, class: "muted" }, svg).textContent = "f(0)";
      let d = "";
      for (let i = 0; i <= 100; i++) { const a = i / 100; d += (i ? "L" : "M") + X(a) + " " + Y(f(a, k, c)); }
      S("path", { d, fill: "none", stroke: "var(--ink)", "stroke-width": 2.5 }, svg);
      const g0 = df(0, k, c);
      S("line", { x1: X(0), y1: Y(0), x2: X(1), y2: Y(g0), stroke: "var(--accent)", "stroke-width": 2, "stroke-dasharray": "6 4" }, svg);
      let ig = 0;
      for (let i = 0; i < n; i++) { const a = (i + 0.5) / n; const g = df(a, k, c); ig += g / n; S("circle", { cx: X(a), cy: Y(f(a, k, c)), r: 4, fill: "var(--purple)" }, svg); }
      S("circle", { cx: X(1), cy: Y(g0), r: 5, fill: "var(--accent)" }, svg);
      S("text", { x: X(1) + 8, y: Y(g0) + 4, "font-size": 12, fill: "var(--accent)", "font-weight": 700 }, svg).textContent = "AtP";
      S("circle", { cx: X(1), cy: Y(1), r: 5, fill: "var(--ink)" }, svg);
      S("text", { x: X(1) + 8, y: Y(1) + 4, "font-size": 12, "font-weight": 700 }, svg).textContent = "true";
      const rows = [["True effect", 1, "var(--ink-2)"], ["Attribution patching", g0, "var(--accent)"], [`Integrated gradients (${n})`, ig, "var(--purple)"]];
      document.getElementById("tanBars").innerHTML = rows.map(([nm, v, col]) => `<div class="barrow"><span>${nm}</span><div class="track"><div class="fill" style="left:0;width:${Math.min(100, Math.max(0, v) / 1.6 * 100)}%;background:${col}"></div></div><span class="val">${v.toFixed(2)}</span></div>`).join("") +
        `<div class="readout">${g0 < 0.3 ? "The slope at α = 0 is almost flat, so attribution patching calls this component unimportant — a false negative. Integrated gradients samples the steep part of the curve and gets much closer." : g0 > 1.7 ? "The slope at α = 0 is very steep, so attribution patching overestimates the effect." : "The curve is fairly gentle, so the straight-line estimate is reasonable."}</div>`;
    }
    [kEl, cEl, nEl].forEach(e => e.oninput = draw);
    draw(); MI.onTheme(draw);
  })();
})();
