/* Chapter 10 widgets and generated figures */
(function () {
  const S = MI.svg;

  /* ---------- 1. Patchscope reader (illustrative) ---------- */
  (function () {
    const tbl = document.getElementById("psTable");
    const sl = document.getElementById("psLayer"), out = document.getElementById("psLayerOut");
    if (!tbl) return;
    // Illustrative readouts for the last token of "Alexander the Great" in a 12-layer model.
    const lens = ["Great", "Great", "Great", "Wall", "Wall", "Barrier", "Alexander", "Alexander", "'s", "'s", "was", "was"];
    const ps = [
      "Great", "great, large", "the Great (a title)", "Alexander the Great", "Alexander the Great, king",
      "Alexander the Great, king of Macedon", "Alexander the Great, king of Macedon",
      "Alexander the Great, Macedonian conqueror", "Alexander the Great, conqueror of Persia",
      "a famous ancient king", "a king who", "was"];
    const psQ = [0.15, 0.3, 0.45, 0.75, 0.85, 0.95, 0.95, 0.9, 0.85, 0.6, 0.4, 0.2];
    const lensQ = [0.1, 0.1, 0.1, 0.1, 0.12, 0.15, 0.35, 0.45, 0.5, 0.55, 0.6, 0.6];
    function draw() {
      const L = +sl.value; out.textContent = L;
      tbl.innerHTML = `
        <tr><td>Source</td><td class="ps-out">"Alexander the Great" → hidden state at layer ${L}, last token</td></tr>
        <tr><td>Logit lens</td><td><span class="bar" style="width:${Math.round(lensQ[L] * 140)}px;background:var(--blue)"></span><span class="ps-out">top token: "${lens[L]}"</span></td></tr>
        <tr><td>Patchscope</td><td><span class="bar" style="width:${Math.round(psQ[L] * 140)}px"></span><span class="ps-out">"${ps[L]}"</span></td></tr>
        <tr><td>What to see</td><td style="font-size:13.5px;color:var(--ink-2)">${
          L < 3 ? "Very early: the state still mostly encodes the current token." :
          L < 9 ? "Middle layers: the logit lens reads a guess of the <i>next</i> token, but the Patchscope can decode the <i>entity</i> the model has built." :
          "Late layers: the state turns into a next-token prediction, so both readers mostly see the next word."}</td></tr>`;
    }
    sl.oninput = draw; draw();
  })();

  /* ---------- 2. Developmental stages (schematic) ---------- */
  (function () {
    const svg = document.getElementById("devSvg"), info = document.getElementById("devInfo");
    if (!svg) return;
    const W = 680, H = 280, x0 = 50, x1 = 660, y0 = 30, y1 = 240;
    const bounds = [0, 0.12, 0.3, 0.48, 0.7, 1];
    const names = ["Stage A", "Stage B", "Stage C", "Stage D", "Stage E"];
    const desc = [
      "Stage A — the model learns simple token statistics, such as bigrams (\"after X, Y is common\"). The loss falls fast.",
      "Stage B — the model starts using position and short-range context. The learning coefficient rises as the model adds structure.",
      "Stage C — heads that attend to the previous token form. This sets up the next step.",
      "Stage D — induction heads form, using the previous-token heads (Chapter 4). This is the in-context-learning bump.",
      "Stage E — refinement. Some structure simplifies, and the learning coefficient can even drop while the loss keeps falling slowly."];
    const X = t => x0 + t * (x1 - x0);
    let sel = -1;
    function loss(t) { return 0.2 + 0.7 * Math.exp(-6 * t) - 0.05 * (1 / (1 + Math.exp(-(t - 0.5) * 40))); }
    function llc(t) {
      let v = 0.1 + 0.25 * (1 / (1 + Math.exp(-(t - 0.12) * 40)));
      v += 0.2 * (1 / (1 + Math.exp(-(t - 0.3) * 40)));
      v += 0.3 * (1 / (1 + Math.exp(-(t - 0.48) * 40)));
      v -= 0.15 * (1 / (1 + Math.exp(-(t - 0.72) * 30)));
      return v;
    }
    function draw() {
      svg.innerHTML = "";
      for (let i = 0; i < 5; i++) {
        const r = S("rect", { x: X(bounds[i]), y: y0, width: X(bounds[i + 1]) - X(bounds[i]), height: y1 - y0, fill: i === sel ? "var(--accent-soft)" : (i % 2 ? "var(--bg-2)" : "var(--panel)"), stroke: "var(--rule)", style: "cursor:pointer" }, svg);
        r.onclick = () => { sel = i; info.textContent = desc[i]; draw(); };
        const t = S("text", { x: (X(bounds[i]) + X(bounds[i + 1])) / 2, y: y0 + 16, "text-anchor": "middle", "font-size": 11.5, style: "pointer-events:none", fill: i === sel ? "var(--accent)" : "var(--ink-3)" }, svg);
        t.textContent = names[i];
      }
      const path = (f, lo, hi) => {
        let d = "";
        for (let k = 0; k <= 200; k++) { const t = k / 200, v = (f(t) - lo) / (hi - lo); d += (k ? "L" : "M") + X(t).toFixed(1) + "," + (y1 - 10 - v * (y1 - y0 - 40)).toFixed(1); }
        return d;
      };
      S("path", { d: path(loss, 0, 0.95), fill: "none", stroke: "var(--blue)", "stroke-width": 2.5, style: "pointer-events:none" }, svg);
      S("path", { d: path(llc, 0, 0.75), fill: "none", stroke: "var(--accent)", "stroke-width": 2.5, style: "pointer-events:none" }, svg);
      S("line", { x1: x0, y1: y1, x2: x1, y2: y1, stroke: "var(--ink-3)" }, svg);
      S("text", { x: (x0 + x1) / 2, y: y1 + 28, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "training steps →";
      S("text", { x: x1 - 6, y: 70, "text-anchor": "end", "font-size": 12, fill: "var(--accent)", style: "pointer-events:none" }, svg).textContent = "LLC estimate";
      S("text", { x: x0 + 70, y: 70, "font-size": 12, fill: "var(--blue)", style: "pointer-events:none" }, svg).textContent = "loss";
    }
    draw();
  })();

  /* ---------- 3. Maze + cheese vector (toy) ---------- */
  (function () {
    const svg = document.getElementById("mazeSvg");
    if (!svg) return;
    const sl = document.getElementById("mazeAlpha"), so = document.getElementById("mazeAlphaOut"), out = document.getElementById("mazeOut");
    // 0 = open, 1 = wall. 11x11 grid.
    const M = [
      "00000100000",
      "01110101110",
      "01000001000",
      "01011111010",
      "00010000010",
      "11010111011",
      "00000100000",
      "01111101110",
      "01000001000",
      "01011101011",
      "00010000000"].map(r => r.split("").map(Number));
    const N = 11, C = 400 / N;
    const start = [10, 0];          // bottom-left
    const corner = [0, 10];         // top-right
    let cheese = [6, 2];
    const open = (r, c) => r >= 0 && c >= 0 && r < N && c < N && !M[r][c];
    function bfs(t) {
      const d = Array.from({ length: N }, () => Array(N).fill(Infinity));
      const q = [t]; d[t[0]][t[1]] = 0;
      while (q.length) {
        const [r, c] = q.shift();
        for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const a = r + dr, b = c + dc;
          if (open(a, b) && d[a][b] === Infinity) { d[a][b] = d[r][c] + 1; q.push([a, b]); }
        }
      }
      return d;
    }
    function run(alpha) {
      const dC = bfs(cheese), dK = bfs(corner);
      const wC = 1 - alpha, wK = 0.45;
      const val = (r, c) => -(wC * dC[r][c] + wK * dK[r][c]);
      let p = start.slice(); const path = [p.slice()], seen = new Set([p.join()]);
      for (let s = 0; s < 60; s++) {
        if (wC > 0.5 && p[0] === cheese[0] && p[1] === cheese[1]) break;
        let best = null, bv = -Infinity;
        for (const [dr, dc] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
          const a = p[0] + dr, b = p[1] + dc;
          if (!open(a, b) || seen.has(a + "," + b)) continue;
          const v = val(a, b);
          if (v > bv) { bv = v; best = [a, b]; }
        }
        if (!best) break;
        p = best; path.push(p.slice()); seen.add(p.join());
        if (p[0] === corner[0] && p[1] === corner[1]) break;
      }
      return path;
    }
    function draw() {
      const alpha = +sl.value; so.textContent = Math.round(alpha * 100) + "%";
      svg.innerHTML = "";
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const cell = S("rect", { x: c * C, y: r * C, width: C, height: C, class: "cell", fill: M[r][c] ? "var(--ink-2)" : "var(--panel)", stroke: "var(--rule)" }, svg);
        if (!M[r][c]) cell.onclick = () => { if (r === start[0] && c === start[1]) return; cheese = [r, c]; draw(); };
      }
      S("rect", { x: corner[1] * C + 2, y: corner[0] * C + 2, width: C - 4, height: C - 4, fill: "none", stroke: "var(--purple)", "stroke-width": 2.5, "stroke-dasharray": "4 3" }, svg);
      const path = run(alpha);
      const d = path.map((p, i) => (i ? "L" : "M") + (p[1] * C + C / 2) + "," + (p[0] * C + C / 2)).join("");
      S("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 4, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: 0.85 }, svg);
      // cheese
      const cx = cheese[1] * C + C / 2, cy = cheese[0] * C + C / 2;
      S("path", { d: `M${cx - 12},${cy + 9} L${cx + 12},${cy + 9} L${cx + 12},${cy - 2} L${cx - 12},${cy - 9} Z`, fill: "var(--yellow)", stroke: "var(--ink-2)", "stroke-width": 1 }, svg);
      // mouse
      const end = path[path.length - 1];
      S("circle", { cx: start[1] * C + C / 2, cy: start[0] * C + C / 2, r: 8, fill: "var(--blue)" }, svg);
      S("circle", { cx: end[1] * C + C / 2, cy: end[0] * C + C / 2, r: 10, fill: "var(--accent)", stroke: "#fff", "stroke-width": 2 }, svg);
      const atCheese = end[0] === cheese[0] && end[1] === cheese[1];
      const atCorner = end[0] === corner[0] && end[1] === corner[1];
      out.textContent = (atCheese ? "Agent reaches the cheese." : atCorner ? "Agent ignores the cheese and goes to the top-right corner (purple)." : "Agent wanders between its two goals.") +
        `  Toy policy: value = −[(1−α)·dist(cheese) + 0.45·dist(top-right)], α = ${alpha.toFixed(2)}.`;
    }
    sl.oninput = draw; draw();
  })();

  /* ---------- 4. Attention sink ---------- */
  (function () {
    const svg = document.getElementById("sinkSvg");
    if (!svg) return;
    const out = document.getElementById("sinkOut"), rm = document.getElementById("sinkRemove");
    const seg = document.getElementById("sinkHead");
    let head = "content";
    seg.onclick = e => { const b = e.target.closest("button"); if (!b) return; head = b.dataset.v; [...seg.children].forEach(x => x.classList.toggle("on", x === b)); draw(); };
    rm.onchange = draw;
    const toks = ["<bos>", "The", "cat", "sat", "on", "the", "mat", "."];
    function pattern() {
      const n = toks.length, A = [];
      const start = rm.checked ? 1 : 0;
      for (let i = 0; i < n; i++) {
        const row = Array(n).fill(0);
        if (i < start) { A.push(row); continue; }
        let w = [];
        for (let j = start; j <= i; j++) {
          let s;
          if (head === "content") s = (j === i - 1 ? 3.2 : 0) + (j === 0 ? 1.2 : 0) + (j === i ? 0.4 : 0);
          else s = (j === 0 ? 4.5 : 0) + 0.15 * Math.sin(3 * i + 5 * j);
          w.push([j, s]);
        }
        const m = Math.max(...w.map(x => x[1])), z = w.reduce((a, x) => a + Math.exp(x[1] - m), 0);
        w.forEach(([j, s]) => (row[j] = Math.exp(s - m) / z));
        A.push(row);
      }
      return A;
    }
    function draw() {
      svg.innerHTML = "";
      const A = pattern(), n = toks.length, cs = 34, ox = 90, oy = 50;
      toks.forEach((t, j) => {
        const tx = S("text", { x: ox + j * cs + cs / 2, y: oy - 10, "text-anchor": "middle", "font-size": 11.5, class: "muted" }, svg); tx.textContent = t;
        const ty = S("text", { x: ox - 8, y: oy + j * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 11.5, class: "muted" }, svg); ty.textContent = t;
      });
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const removed = rm.checked && (i === 0 || j === 0);
        S("rect", { x: ox + j * cs, y: oy + i * cs, width: cs - 2, height: cs - 2, rx: 3, fill: j > i || removed ? "var(--bg-2)" : MI.diverge(A[i][j]), stroke: removed ? "var(--rule)" : "none" }, svg);
        if (j <= i && !removed && A[i][j] > 0.2) {
          const t = S("text", { x: ox + j * cs + cs / 2 - 1, y: oy + i * cs + cs / 2 + 3, "text-anchor": "middle", "font-size": 9.5, fill: A[i][j] > 0.6 ? "#fff" : "var(--ink)" }, svg);
          t.textContent = A[i][j].toFixed(2).replace(/^0/, "");
        }
      }
      S("text", { x: ox + n * cs / 2, y: oy + n * cs + 22, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "key (attended token) →";
      const tx = S("text", { x: 22, y: oy + n * cs / 2, "font-size": 12, class: "muted", transform: `rotate(-90 22 ${oy + n * cs / 2})`, "text-anchor": "middle" }, svg); tx.textContent = "query →";
      const avgSink = A.slice(1).reduce((a, r) => a + r[0], 0) / (n - 1);
      out.textContent = head === "content"
        ? (rm.checked ? "Without the sink, the content head still attends to the previous token. Its job does not need the sink much." : `Content head: mostly the previous token. Some spare attention still parks on <bos> (avg ${avgSink.toFixed(2)}).`)
        : (rm.checked ? "Without a sink, the dormant head has nowhere harmless to park. Its attention spreads over real tokens and mixes in noise. This is why streaming models keep the first tokens." : `Dormant head: on average ${(avgSink * 100).toFixed(0)}% of attention goes to <bos>. The head "does nothing" by parking attention on a harmless token.`);
    }
    draw();
  })();

  /* ---------- static figure: chess board (Fig 10.4) ---------- */
  (function () {
    const g = document.getElementById("chessBoard");
    if (!g) return;
    const ox = 20, oy = 20, c = 36;
    for (let r = 0; r < 8; r++) for (let f = 0; f < 8; f++)
      S("rect", { x: ox + f * c, y: oy + r * c, width: c, height: c, fill: (r + f) % 2 ? "var(--bg-2)" : "var(--panel)", stroke: "var(--rule)", "stroke-width": 0.5 }, g);
    "abcdefgh".split("").forEach((ch, i) => { S("text", { x: ox + i * c + c / 2, y: oy + 8 * c + 14, "text-anchor": "middle", "font-size": 10, class: "muted" }, g).textContent = ch; });
    const sq = (f, r) => [ox + f * c + c / 2, oy + (7 - r) * c + c / 2];
    // glow on third-ply target square (g7 → f, r index: g=6, 7 → r=6)
    const [gx, gy] = sq(5, 6);
    S("rect", { x: gx - c / 2, y: gy - c / 2, width: c, height: c, fill: "var(--accent)", opacity: 0.35 }, g);
    S("rect", { x: gx - c / 2 + 1, y: gy - c / 2 + 1, width: c - 2, height: c - 2, fill: "none", stroke: "var(--accent)", "stroke-width": 2.5 }, g);
    const arrow = (a, b, col, mk, dash) => S("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: col, "stroke-width": 4, "marker-end": `url(#${mk})`, "stroke-dasharray": dash || "", opacity: 0.9 }, g);
    arrow(sq(3, 0), sq(3, 4), "var(--blue)", "ca");           // d1 → d5
    arrow(sq(6, 7), sq(5, 5), "var(--ink-3)", "cb");          // g8 → f6 (knight)
    arrow(sq(3, 4), sq(5, 6), "var(--accent)", "cc");         // d5 → f7 (diagonal)
    const a = sq(5, 6), b = sq(3, 3);
    S("path", { d: `M${a[0] - 6},${a[1] + 12} C ${a[0] + 10},${b[1] + 40}, ${b[0] + 40},${b[1] + 40}, ${b[0] + 6},${b[1] + 14}`, fill: "none", stroke: "var(--purple)", "stroke-width": 2.5, "stroke-dasharray": "5 4", "marker-end": "url(#cd)" }, g);
    // pieces (letters)
    [[3, 0, "♕"], [6, 7, "♞"], [4, 0, "♔"], [4, 7, "♚"]].forEach(([f, r, p]) => {
      const [x, y] = sq(f, r);
      S("text", { x, y: y + 9, "text-anchor": "middle", "font-size": 24 }, g).textContent = p;
    });
  })();

  /* ---------- static figure: entropy neuron bars (Fig 10.6) ---------- */
  (function () {
    const g = document.getElementById("entBars");
    if (!g) return;
    const logits = [3.0, 2.0, 1.4, 0.8, 0.3];
    const sm = (l, T) => { const e = l.map(v => Math.exp(v / T)); const z = e.reduce((a, b) => a + b, 0); return e.map(v => v / z); };
    const draw = (ox, p, col) => p.forEach((v, i) => {
      const h = v * 230;
      S("rect", { x: ox + i * 44, y: 190 - h, width: 32, height: h, rx: 3, fill: col }, g);
      S("text", { x: ox + i * 44 + 16, y: 184 - h, "text-anchor": "middle", "font-size": 10.5, class: "muted" }, g).textContent = (v * 100).toFixed(0) + "%";
      S("text", { x: ox + i * 44 + 16, y: 206, "text-anchor": "middle", "font-size": 11 }, g).textContent = "t" + (i + 1);
    });
    draw(55, sm(logits, 1), "var(--blue)");
    draw(315, sm(logits, 2.2), "var(--accent)");
  })();

  /* ---------- static figure: thought anchors (Fig 10.7) ---------- */
  (function () {
    const g = document.getElementById("anchorBars");
    if (!g) return;
    const rows = [
      ["Let me plan: first find the total, then divide.", 0.82, "plan"],
      ["Total = 12 × 7 = 84.", 0.12, ""],
      ["Divide by 4 gives 21.", 0.1, ""],
      ["Wait — the question asks per person, not per group.", 0.9, "backtrack"],
      ["So there are 3 people per group.", 0.22, ""],
      ["21 / 3 = 7.", 0.08, ""],
      ["Answer: 7.", 0.05, ""]];
    rows.forEach(([t, v, k], i) => {
      const y = 22 + i * 33;
      S("text", { x: 10, y: y + 13, "font-size": 12.5, fill: k ? "var(--ink)" : "var(--ink-2)", "font-weight": k ? 700 : 400 }, g).textContent = t;
      S("rect", { x: 420, y, width: v * 270, height: 18, rx: 3, fill: k ? "var(--accent)" : "var(--ink-3)", opacity: k ? 1 : 0.5 }, g);
      if (k) S("text", { x: 420 + v * 270 - 6, y: y + 13, "text-anchor": "end", "font-size": 11, fill: "#fff", "font-weight": 700 }, g).textContent = k;
    });
    S("text", { x: 420, y: 252, "font-size": 11, class: "muted" }, g).textContent = "counterfactual importance (schematic) →";
  })();

  /* ---------- static figure: MLP vs KAN (Fig 10.10) ---------- */
  (function () {
    const g = document.getElementById("kanFig");
    if (!g) return;
    const inL = [[60, 70], [60, 150]], outL = [[280, 110]];
    const inR = [[420, 70], [420, 150]], outR = [[640, 110]];
    // MLP: edges with numbers, nodes with ReLU
    const hidL = [[170, 50], [170, 110], [170, 170]];
    const ws = ["0.8", "−1.2", "0.4", "1.5", "−0.3", "0.9"];
    let k = 0;
    inL.forEach(a => hidL.forEach(b => {
      S("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "var(--ink-3)", "stroke-width": 1.2 }, g);
    }));
    hidL.forEach(b => outL.forEach(o => S("line", { x1: b[0], y1: b[1], x2: o[0], y2: o[1], stroke: "var(--ink-3)", "stroke-width": 1.2 }, g)));
    [[110, 52], [112, 96], [100, 128], [120, 160]].forEach(([x, y]) => { S("text", { x, y, "font-size": 10, fill: "var(--blue)", "text-anchor": "middle" }, g).textContent = ws[k++]; });
    inL.forEach(([x, y]) => S("circle", { cx: x, cy: y, r: 12, fill: "var(--panel)", stroke: "var(--ink-2)" }, g));
    hidL.forEach(([x, y]) => {
      S("circle", { cx: x, cy: y, r: 15, fill: "var(--blue-soft)", stroke: "var(--blue)" }, g);
      S("path", { d: `M${x - 8},${y + 4} L${x},${y + 4} L${x + 8},${y - 5}`, fill: "none", stroke: "var(--blue)", "stroke-width": 1.8 }, g);
    });
    outL.forEach(([x, y]) => S("circle", { cx: x, cy: y, r: 12, fill: "var(--panel)", stroke: "var(--ink-2)" }, g));
    S("text", { x: 170, y: 205, "text-anchor": "middle", "font-size": 11, class: "muted" }, g).textContent = "numbers on edges · fixed ReLU on nodes";
    // KAN: edges with mini curves, nodes are sums
    const hidR = [[530, 70], [530, 150]];
    const curves = [
      t => Math.sin(t * 6.28), t => t * t * 2 - 1, t => Math.exp(t * 2) / 4 - 1, t => -Math.sin(t * 3.14),
      t => Math.cos(t * 6.28), t => 1 - 2 * t];
    let ci = 0;
    const edge = (a, b) => {
      S("line", { x1: a[0], y1: a[1], x2: b[0], y2: b[1], stroke: "var(--rule)", "stroke-width": 1.2 }, g);
      const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2, f = curves[ci++ % curves.length];
      S("rect", { x: mx - 17, y: my - 12, width: 34, height: 24, rx: 4, fill: "var(--panel)", stroke: "var(--accent)" }, g);
      let d = "";
      for (let s = 0; s <= 20; s++) { const t = s / 20; d += (s ? "L" : "M") + (mx - 13 + t * 26).toFixed(1) + "," + (my - Math.max(-1, Math.min(1, f(t))) * 8).toFixed(1); }
      S("path", { d, fill: "none", stroke: "var(--accent)", "stroke-width": 1.6 }, g);
    };
    inR.forEach(a => hidR.forEach(b => edge(a, b)));
    hidR.forEach(b => outR.forEach(o => edge(b, o)));
    inR.forEach(([x, y]) => S("circle", { cx: x, cy: y, r: 12, fill: "var(--panel)", stroke: "var(--ink-2)" }, g));
    hidR.concat(outR).forEach(([x, y]) => { S("circle", { cx: x, cy: y, r: 13, fill: "var(--accent-soft)", stroke: "var(--accent)" }, g); S("text", { x, y: y + 5, "text-anchor": "middle", "font-size": 14, fill: "var(--accent)" }, g).textContent = "+"; });
    S("text", { x: 530, y: 205, "text-anchor": "middle", "font-size": 11, class: "muted" }, g).textContent = "learned curve on every edge · nodes only add";
  })();
})();
