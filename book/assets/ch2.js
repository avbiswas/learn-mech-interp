/* Chapter 2 — figures and widgets (vision era). Needs app.js (window.MI). */
(function () {
  const S = MI.svg;
  const $ = id => document.getElementById(id);

  /* ---------------- static figure helpers ---------------- */

  // Figure 2.5: sentiment histogram humps
  (function () {
    const hump = (mu, sig, h) => {
      let d = "M60 160";
      for (let x = 60; x <= 680; x += 5) d += ` L${x} ${(160 - h * Math.exp(-((x - mu) ** 2) / (2 * sig * sig))).toFixed(1)}`;
      return d + " L680 160 Z";
    };
    $("sentNeg").setAttribute("d", hump(220, 62, 112));
    $("sentPos").setAttribute("d", hump(520, 62, 112));
  })();

  // Figure 2.3: four small grids (activation, mask, concept, overlap)
  const LAMP10 = (() => {
    const m = [];
    for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++) {
      const shade = r >= 1 && r <= 3 && c >= 3 && c <= 6;
      const stem = r >= 4 && r <= 7 && c >= 4 && c <= 5;
      const base = r === 8 && c >= 3 && c <= 6;
      m.push(shade || stem || base ? 1 : 0);
    }
    return m;
  })();
  function drawND() {
    const g = $("ndLowres");
    if (!g) return;
    g.innerHTML = "";
    const act = [];
    for (let r = 0; r < 10; r++) for (let c = 0; c < 10; c++)
      act.push(Math.exp(-(((r - 2.6) ** 2) / 5 + ((c - 4.5) ** 2) / 5)) + 0.25 * Math.exp(-(((r - 6) ** 2) + ((c - 5) ** 2)) / 3));
    const sorted = [...act].sort((a, b) => b - a), thr = sorted[13];
    const mask = act.map(v => (v > thr ? 1 : 0));
    const grids = [
      { x: 20, f: i => MI.diverge(act[i]) },
      { x: 175, f: i => (mask[i] ? "var(--accent)" : "var(--bg-2)") },
      { x: 330, f: i => (LAMP10[i] ? "var(--blue)" : "var(--bg-2)") },
      { x: 485, f: i => (mask[i] && LAMP10[i] ? "var(--green)" : mask[i] ? "var(--accent)" : LAMP10[i] ? "var(--blue)" : "var(--bg-2)") },
    ];
    grids.forEach(G => {
      for (let i = 0; i < 100; i++)
        S("rect", { x: G.x + (i % 10) * 10, y: 55 + Math.floor(i / 10) * 10, width: 9.4, height: 9.4, fill: G.f(i), opacity: G.x === 485 && !mask[i] && LAMP10[i] ? 0.55 : 1 }, g);
      S("rect", { x: G.x - 1, y: 54, width: 101, height: 101, fill: "none", stroke: "var(--rule)" }, g);
    });
  }
  drawND();
  MI.onTheme(drawND);

  // Figure 2.6: InceptionV1 layer stack
  (function () {
    const g = $("incStack");
    const L = ["conv2d0", "conv2d1", "conv2d2", "mixed3a", "mixed3b", "mixed4a", "mixed4b", "mixed4c", "mixed4d", "mixed4e", "mixed5a", "mixed5b"];
    const tips = {
      conv2d0: "64 neurons: Gabor filters (44%) and color contrast (42%).",
      conv2d1: "64 neurons: low frequency, Gabor-like, color contrast, multicolor.",
      conv2d2: "192 neurons: lines, shifted lines, textures, tiny curves, corners.",
      mixed3a: "256 neurons: early curves, 15 high-low frequency detectors, black & white, small circles.",
      mixed3b: "480 neurons: curve detectors (e.g. 3b:379), boundary detectors, circles, spirals, proto-heads.",
      mixed4a: "Curve family (12 neurons) and the start of object parts.",
      mixed4b: "Parts such as windows, car bodies, wheels, eyes.",
      mixed4c: "The car neuron: windows on top, wheels at the bottom.",
      mixed4d: "Objects; the car feature spreads into dog neurons (superposition).",
      mixed4e: "Objects; polysemantic neuron 4e:55 (cat faces, car fronts, cat legs).",
      mixed5a: "High-level, class-like features.",
      mixed5b: "Last conv layer; features close to the output classes. Weight banding seen here.",
    };
    const groups = [
      { a: 0, b: 1, c: "var(--green)", t: ["Gabor filters,", "color contrast"], lvl: 0 },
      { a: 2, b: 2, c: "var(--green)", t: ["lines, tiny curves,", "textures"], lvl: 0 },
      { a: 3, b: 3, c: "var(--blue)", t: ["early curves,", "high-low frequency"], lvl: 1 },
      { a: 4, b: 4, c: "var(--blue)", t: ["curves, boundaries,", "proto-heads"], lvl: 0 },
      { a: 5, b: 7, c: "var(--purple)", t: ["parts: eyes, wheels,", "windows, fur"], lvl: 1 },
      { a: 8, b: 9, c: "var(--accent)", t: ["objects: dog heads,", "cars; 4e:55"], lvl: 0 },
      { a: 10, b: 11, c: "var(--red)", t: ["class-level", "features"], lvl: 1 },
    ];
    const X = i => 24 + i * 76, W = 68;
    S("text", { x: 24, y: 24, "font-size": 12, class: "muted" }, g).textContent = "image →";
    S("text", { x: 936, y: 24, "font-size": 12, class: "muted", "text-anchor": "end" }, g).textContent = "→ 1000 ImageNet classes";
    L.forEach((name, i) => {
      const grp = groups.find(q => i >= q.a && i <= q.b);
      const h = 34 + i * 2.5;
      const r = S("rect", { x: X(i), y: 80 - h / 2, width: W, height: h, rx: 7, fill: "var(--panel)", stroke: grp.c, "stroke-width": 1.6 }, g);
      S("title", {}, r).textContent = name + ": " + tips[name];
      const t = S("text", { x: X(i) + W / 2, y: 84, "text-anchor": "middle", "font-size": 11.5, "font-family": "var(--mono)" }, g);
      t.textContent = name;
      S("title", {}, t).textContent = name + ": " + tips[name];
    });
    groups.forEach(q => {
      const x0 = X(q.a), x1 = X(q.b) + W, cx = (x0 + x1) / 2, y = 128 + q.lvl * 52;
      S("path", { d: `M${x0 + 4} 112 L${x0 + 4} 118 L${x1 - 4} 118 L${x1 - 4} 112`, fill: "none", stroke: q.c, "stroke-width": 1.5 }, g);
      if (q.lvl) S("line", { x1: cx, y1: 118, x2: cx, y2: y - 10, stroke: q.c, "stroke-dasharray": "2 3" }, g);
      q.t.forEach((s, k) => {
        S("text", { x: cx, y: y + 6 + k * 15, "text-anchor": "middle", "font-size": 12, fill: q.c, "font-weight": k ? 400 : 700 }, g).textContent = s;
      });
    });
    S("text", { x: 480, y: 244, "text-anchor": "middle", "font-size": 11.5, class: "muted" }, g).textContent = "Block height grows only to suggest depth. Hover a layer name for details from the Circuits papers.";
  })();

  // Figure 2.7: curve circuit weight grids
  function drawCC() {
    const g = $("ccGrids");
    if (!g) return;
    g.innerHTML = "";
    const cell = 21;
    const arcW = (r, c) => {
      const d = Math.hypot(c - 2, r - 4.0);
      return Math.exp(-((d - 2.4) ** 2) / 0.45) - (d < 1.5 ? 0.35 : 0);
    };
    const rows = [
      { y: 18, label: ["early curve detector", "same orientation"], sign: 1, flip: false },
      { y: 148, label: ["early curve detector", "opposite orientation"], sign: -0.75, flip: true },
    ];
    rows.forEach(R => {
      // icon + label
      const iy = R.y + 52;
      S("circle", { cx: 60, cy: iy, r: 26, fill: "var(--green-soft)", stroke: "var(--green)" }, g);
      S("path", { d: R.flip ? `M44 ${iy - 6} A 18 18 0 0 0 76 ${iy - 6}` : `M44 ${iy + 6} A 18 18 0 0 1 76 ${iy + 6}`, fill: "none", stroke: "var(--green)", "stroke-width": 3 }, g);
      S("text", { x: 100, y: iy - 6, "font-size": 12.5, "font-weight": 700 }, g).textContent = R.label[0];
      S("text", { x: 100, y: iy + 12, "font-size": 12.5, class: "muted" }, g).textContent = R.label[1] + " (mixed3a)";
      S("line", { x1: 270, y1: iy, x2: 340, y2: iy, stroke: "var(--ink-3)", "stroke-width": 1.5, "marker-end": "url(#a25)" }, g);
      for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
        const w = R.sign > 0 ? arcW(r, c) : -0.8 * Math.max(0, arcW(r, c)) - 0.08;
        S("rect", { x: 350 + c * cell, y: R.y + r * cell, width: cell - 2, height: cell - 2, rx: 3, fill: MI.diverge(w) }, g);
      }
      S("text", { x: 350 + 2.5 * cell, y: R.y + 5 * cell + 12, "text-anchor": "middle", "font-size": 11, class: "muted" }, g).textContent = "5×5 weights";
    });
  }
  drawCC();
  MI.onTheme(drawCC);

  // Figure 2.9: 5x5 car neuron grid
  (function () {
    const g = $("carGrid");
    const cs = 12, x0 = 300, y0 = 160;
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      const fill = r <= 1 ? "var(--green)" : r >= 3 ? "var(--ink-2)" : "var(--bg-2)";
      S("rect", { x: x0 + c * cs, y: y0 + r * cs, width: cs - 1.5, height: cs - 1.5, rx: 2, fill, opacity: r === 2 ? 1 : 0.75 }, g);
    }
    S("text", { x: x0 + 68, y: y0 + 16, "font-size": 11.5, fill: "var(--green)" }, g).textContent = "← windows";
    S("text", { x: x0 + 68, y: y0 + 52, "font-size": 11.5, class: "muted" }, g).textContent = "← wheels";
  })();

  // Figure 2.11: GPU icons
  (function () {
    const g1 = $("gpu1"), g2 = $("gpu2");
    for (let i = 0; i < 7; i++) {
      const x = 92 + i * 30, y = 52;
      const grp = S("g", {}, g1);
      S("rect", { x, y, width: 26, height: 26, rx: 4, fill: "var(--panel)", stroke: "var(--rule)" }, grp);
      const ang = i * 26;
      const cp = S("clipPath", { id: "gp" + i }, grp);
      S("rect", { x, y, width: 26, height: 26, rx: 4 }, cp);
      const st = S("g", { "clip-path": `url(#gp${i})`, transform: `rotate(${ang} ${x + 13} ${y + 13})` }, grp);
      for (let k = -2; k <= 2; k++) S("rect", { x: x - 6, y: y + 11 + k * 7, width: 38, height: 3.2, fill: k % 2 ? "var(--ink)" : "var(--ink-3)" }, st);
    }
    const cols = ["#e76f51", "#2a9d8f", "#e9c46a", "#8ab17d", "#9d4edd", "#f4a261", "#4361ee"];
    cols.forEach((c, i) => {
      const x = 92 + i * 30, y = 132;
      S("rect", { x, y, width: 26, height: 26, rx: 4, fill: "var(--panel)", stroke: "var(--rule)" }, g2);
      S("circle", { cx: x + 13, cy: y + 13, r: 9, fill: c, opacity: 0.75 }, g2);
    });
  })();

  // Figure 2.12: CLIP comparison grid
  (function () {
    const g = $("clipGrid");
    const cols = [
      { x: 300, h: ["Human brain cell", "(\"Halle Berry\", 2005)"], on: [1, 1, 1] },
      { x: 465, h: ["CLIP neuron 244", "(\"Spider-Man\")"], on: [1, 1, 1] },
      { x: 630, h: ["InceptionV1 neuron 483", "(generic person)"], on: [1, 0.25, 0] },
    ];
    const rows = ["Photo", "Drawing / comic", "Written name"];
    cols.forEach(c => c.h.forEach((t, k) =>
      S("text", { x: c.x, y: 26 + k * 16, "text-anchor": "middle", "font-size": 12.5, "font-weight": k ? 400 : 700, class: k ? "muted" : "" }, g).textContent = t));
    rows.forEach((r, i) => {
      const y = 92 + i * 52;
      S("line", { x1: 20, y1: y + 26, x2: 700, y2: y + 26, stroke: "var(--rule)" }, g);
      S("text", { x: 24, y: y + 5, "font-size": 13 }, g).textContent = r;
      cols.forEach(c => {
        const v = c.on[i];
        S("circle", { cx: c.x, cy: y, r: 15, fill: v ? "var(--accent)" : "var(--bg-2)", "fill-opacity": v ? 0.25 + 0.75 * v : 1, stroke: v ? "var(--accent)" : "var(--rule)", "stroke-width": 1.5 }, g);
        S("text", { x: c.x + 24, y: y + 4, "font-size": 11.5, class: "muted" }, g).textContent = v >= 1 ? "fires" : v > 0 ? "weak" : "—";
      });
    });
  })();

  /* ---------------- Widget: sanity check ---------------- */
  (function () {
    const N = 48;
    const rng = MI.rng(7);
    // input image: a bird on a branch with some grass texture
    const img = new Float32Array(N * N), obj = new Float32Array(N * N);
    for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
      const body = ((x - 22) / 11) ** 2 + ((y - 24) / 7.5) ** 2 < 1;
      const head = (x - 32) ** 2 + (y - 16) ** 2 < 30;
      const tail = x > 6 && x < 14 && Math.abs(y - 22 - (14 - x) * 0.4) < 2.2;
      const branch = y > 33 && y < 36 && x > 2 && x < 44;
      const grass = y > 40 && (x * 7 + y * 3) % 6 < 2;
      const o = body || head || tail;
      obj[y * N + x] = o ? 1 : 0;
      img[y * N + x] = o ? 0.15 : branch ? 0.35 : grass ? 0.55 : 0.92;
    }
    const sob = new Float32Array(N * N);
    for (let y = 1; y < N - 1; y++) for (let x = 1; x < N - 1; x++) {
      const p = (a, b) => img[(y + b) * N + x + a];
      const gx = p(1, -1) + 2 * p(1, 0) + p(1, 1) - p(-1, -1) - 2 * p(-1, 0) - p(-1, 1);
      const gy = p(-1, 1) + 2 * p(0, 1) + p(1, 1) - p(-1, -1) - 2 * p(0, -1) - p(1, -1);
      sob[y * N + x] = Math.min(1, Math.hypot(gx, gy) / 2.2);
    }
    const blur = (a, rad) => {
      const o = new Float32Array(N * N);
      for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
        let s = 0, n = 0;
        for (let dy = -rad; dy <= rad; dy++) for (let dx = -rad; dx <= rad; dx++) {
          const yy = y + dy, xx = x + dx;
          if (yy >= 0 && yy < N && xx >= 0 && xx < N) { s += a[yy * N + xx]; n++; }
        }
        o[y * N + x] = s / n;
      }
      return o;
    };
    const norm = a => { let m = 0; a.forEach(v => (m = Math.max(m, v))); return a.map(v => v / (m || 1)); };
    const noise = (seed, smooth) => { const r = MI.rng(seed); let a = new Float32Array(N * N).map(() => Math.abs(MI.gauss(r))); if (smooth) a = blur(a, smooth); return norm(a); };
    const objB = norm(blur(obj, 2));
    const n1 = noise(11, 0), n2 = noise(23, 1), c1 = noise(31, 7), c2 = noise(47, 7);
    const coarse = a => { const o = new Float32Array(N * N); for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) { const by = Math.floor(y / 8) * 8 + 4, bx = Math.floor(x / 8) * 8 + 4; o[y * N + x] = a[Math.min(N - 1, by) * N + Math.min(N - 1, bx)]; } return norm(blur(o, 3)); };
    const maps = {
      Gradient: [
        norm(objB.map((v, i) => v * (0.5 + 0.5 * n1[i]) + 0.08 * n2[i])),
        norm(objB.map((v, i) => 0.35 * v * n1[i] + 0.65 * n2[i])),
        n1,
      ],
      "Guided backprop": [
        norm(sob.map((v, i) => v * (0.8 + 0.2 * objB[i]))),
        norm(sob.map((v, i) => v * (0.92 + 0.08 * n2[i]))),
        norm(sob.map((v, i) => v * (0.9 + 0.1 * n1[i]))),
      ],
      "Grad-CAM": [coarse(blur(obj, 4)), coarse(c1), coarse(c2)],
    };
    const corr = (a, b) => {
      let ma = 0, mb = 0; for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
      ma /= a.length; mb /= b.length;
      let s = 0, sa = 0, sb = 0;
      for (let i = 0; i < a.length; i++) { const x = a[i] - ma, y = b[i] - mb; s += x * y; sa += x * x; sb += y * y; }
      return s / Math.sqrt(sa * sb);
    };
    const grid = $("sanityGrid"), out = $("sanityOut");
    let state = 2;
    const paint = (cv, arr, gray) => {
      const ctx = cv.getContext("2d"), im = ctx.createImageData(N, N);
      for (let i = 0; i < N * N; i++) {
        let r, g, b;
        if (gray) { r = g = b = Math.round(arr[i] * 255); }
        else { const m = MI.diverge(arr[i]).match(/\d+/g).map(Number); [r, g, b] = m; }
        im.data.set([r, g, b, 255], i * 4);
      }
      ctx.putImageData(im, 0, 0);
    };
    function build() {
      grid.innerHTML = "";
      const names = ["Trained model", "Top layers random", "All layers random"];
      ["", "Input", "Trained model", names[state] + (state ? "" : " (same)")].forEach(h => MI.el("div", { class: "h", text: h }, grid));
      const lines = [];
      Object.keys(maps).forEach(k => {
        MI.el("div", { class: "rl", text: k }, grid);
        const a = MI.el("canvas", { width: N, height: N, class: "pix" }, grid); paint(a, img, true);
        const b = MI.el("canvas", { width: N, height: N, class: "pix" }, grid); paint(b, maps[k][0]);
        const c = MI.el("canvas", { width: N, height: N, class: "pix" }, grid); paint(c, maps[k][state]);
        const r = corr(maps[k][0], maps[k][state]);
        lines.push(`${k.padEnd(16)} similarity to trained map: ${r.toFixed(2)}${state ? (r > 0.8 ? "   ⚠ barely changed — fails the check" : "   ✓ changed — passes the check") : ""}`);
      });
      out.textContent = lines.join("\n");
      out.style.whiteSpace = "pre";
      out.style.overflowX = "auto";
    }
    $("sanitySeg").addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      state = +b.dataset.v;
      [...$("sanitySeg").children].forEach(x => x.classList.toggle("on", x === b));
      build();
    });
    build();
    MI.onTheme(build);
  })();

  /* ---------------- Widget: feature visualization ---------------- */
  (function () {
    const N = 32;
    const cv = $("fvCanvas"), ctx = cv.getContext("2d");
    const wv = $("fvWeights"), wctx = wv.getContext("2d");
    const pc = $("fvCurve"), pctx = pc.getContext("2d");
    // build the neuron: oriented line kernels along an arc
    const units = [];
    const cx = 16, cy = 23, R = 11;
    for (let t = 200; t <= 340; t += 10) {
      const a = (t * Math.PI) / 180;
      const px = cx + R * Math.cos(a), py = cy + R * Math.sin(a);
      const tx = -Math.sin(a), ty = Math.cos(a);
      const k = [];
      let sum = 0;
      for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
        const ox = Math.round(px) + dx, oy = Math.round(py) + dy;
        const fx = ox - px, fy = oy - py;
        const par = fx * tx + fy * ty, perp = -fx * ty + fy * tx;
        const v = (1 - perp * perp / 1.1) * Math.exp(-perp * perp / 2.2) * Math.exp(-par * par / 8);
        k.push({ x: ox, y: oy, w: v }); sum += v;
      }
      const mean = sum / k.length;
      let n2 = 0; k.forEach(e => { e.w -= mean; n2 += e.w * e.w; });
      k.forEach(e => (e.w /= Math.sqrt(n2)));
      units.push(k);
    }
    let img, hist, running = false, steps = 0;
    const rng = MI.rng(3);
    function reset() {
      img = new Float32Array(N * N).map(() => (rng() * 2 - 1) * 0.85);
      hist = []; steps = 0; draw();
    }
    const at = (x, y) => (x >= 0 && x < N && y >= 0 && y < N ? y * N + x : -1);
    function activation(jx = 0, jy = 0, grad) {
      let total = 0;
      units.forEach(k => {
        let pre = 0;
        k.forEach(e => { const i = at(e.x + jx, e.y + jy); if (i >= 0) pre += e.w * img[i]; });
        const slope = pre > 0 ? 1 : 0.05;
        total += pre > 0 ? pre : 0.05 * pre;
        if (grad) k.forEach(e => { const i = at(e.x + jx, e.y + jy); if (i >= 0) grad[i] += slope * e.w; });
      });
      return total;
    }
    function step() {
      const jit = $("fvJitter").checked, bl = $("fvBlur").checked, dec = $("fvDecay").checked;
      const g = new Float32Array(N * N);
      const jx = jit ? Math.round(rng() * 4 - 2) : 0, jy = jit ? Math.round(rng() * 4 - 2) : 0;
      activation(jx, jy, g);
      for (let i = 0; i < N * N; i++) img[i] += 0.12 * g[i];
      if (bl) {
        const o = new Float32Array(N * N);
        for (let y = 0; y < N; y++) for (let x = 0; x < N; x++) {
          let s = 0, n = 0;
          for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) { const i = at(x + dx, y + dy); if (i >= 0) { s += img[i]; n++; } }
          o[y * N + x] = 0.75 * img[y * N + x] + 0.25 * (s / n);
        }
        img = o;
      }
      if (dec) for (let i = 0; i < N * N; i++) img[i] *= 0.975;
      for (let i = 0; i < N * N; i++) img[i] = Math.max(-1, Math.min(1, img[i]));
      steps++;
    }
    function draw() {
      const im = ctx.createImageData(N, N);
      for (let i = 0; i < N * N; i++) { const v = Math.round((img[i] * 0.5 + 0.5) * 255); im.data.set([v, v, v, 255], i * 4); }
      ctx.putImageData(im, 0, 0);
      const a = activation();
      hist.push(a);
      // weights picture
      const w = new Float32Array(N * N);
      units.forEach(k => k.forEach(e => { const i = at(e.x, e.y); if (i >= 0) w[i] += e.w; }));
      let m = 0; w.forEach(v => (m = Math.max(m, Math.abs(v))));
      const wi = wctx.createImageData(N, N);
      for (let i = 0; i < N * N; i++) { const c = MI.diverge(w[i] / m).match(/\d+/g).map(Number); wi.data.set([c[0], c[1], c[2], 255], i * 4); }
      wctx.putImageData(wi, 0, 0);
      // curve
      const W = pc.width, H = pc.height;
      pctx.clearRect(0, 0, W, H);
      pctx.strokeStyle = MI.css("--rule"); pctx.strokeRect(0.5, 0.5, W - 1, H - 1);
      const max = Math.max(5, ...hist);
      pctx.strokeStyle = MI.css("--accent"); pctx.lineWidth = 2; pctx.beginPath();
      hist.forEach((v, i) => { const x = 4 + (i / Math.max(1, 300 / 4)) * (W - 8), y = H - 6 - (Math.max(0, v) / max) * (H - 12); i ? pctx.lineTo(x, y) : pctx.moveTo(x, y); });
      pctx.stroke();
      let noiseE = 0, n = 0;
      const near = new Uint8Array(N * N);
      units.forEach(k => k.forEach(e => { const i = at(e.x, e.y); if (i >= 0) near[i] = 1; }));
      for (let i = 0; i < N * N; i++) if (!near[i]) { noiseE += Math.abs(img[i]); n++; }
      $("fvStat").innerHTML = `step <b>${steps}</b> · neuron activation <b>${a.toFixed(2)}</b> · leftover noise away from the curve <b>${(noiseE / n).toFixed(2)}</b>`;
    }
    function loop() {
      if (!running) return;
      for (let i = 0; i < 4; i++) step();
      draw();
      if (steps >= 300) { running = false; $("fvRun").textContent = "Run"; return; }
      requestAnimationFrame(loop);
    }
    $("fvRun").onclick = () => {
      running = !running;
      $("fvRun").textContent = running ? "Pause" : "Run";
      if (running) { if (steps >= 300) { steps = 0; hist = []; } loop(); }
    };
    $("fvReset").onclick = () => { running = false; $("fvRun").textContent = "Run"; reset(); };
    reset();
    MI.onTheme(draw);
  })();

  /* ---------------- Widget: Network Dissection IoU ---------------- */
  (function () {
    const N = 16, rng = MI.rng(5);
    const act = [], lamp = [], floor = [];
    for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
      act.push(Math.exp(-(((r - 5) ** 2) + ((c - 7.5) ** 2)) / 9) + 0.45 * Math.exp(-(((r - 10.5) ** 2) + ((c - 7.5) ** 2)) / 4) + 0.12 * rng());
      const shade = r >= 3 && r <= 7 && c >= 5 && c <= 10 && Math.abs(c - 7.5) <= 1.5 + (r - 3) * 0.6;
      const stem = r >= 8 && r <= 12 && c >= 7 && c <= 8;
      const base = r === 13 && c >= 5 && c <= 10;
      lamp.push(shade || stem || base ? 1 : 0);
      floor.push(r >= 14 ? 1 : 0);
    }
    const sorted = [...act].sort((a, b) => b - a);
    let concept = "lamp";
    const box = $("ndGrids");
    function iou(mask, m) { let i = 0, u = 0; for (let k = 0; k < N * N; k++) { if (mask[k] && m[k]) i++; if (mask[k] || m[k]) u++; } return u ? i / u : 0; }
    function grid(title, f) {
      const col = MI.el("div", { class: "col" }, box);
      const s = MI.svg("svg", { viewBox: `0 0 ${N * 12} ${N * 12}`, width: 192, height: 192 }, col);
      for (let k = 0; k < N * N; k++) MI.svg("rect", { x: (k % N) * 12, y: Math.floor(k / N) * 12, width: 11, height: 11, rx: 1.5, fill: f(k) }, s);
      MI.el("span", { text: title }, col);
    }
    function draw() {
      const q = +$("ndThr").value;
      $("ndThrOut").textContent = q + "%";
      const keep = Math.max(1, Math.round((q / 100) * N * N));
      const thr = sorted[keep - 1];
      const mask = act.map(v => (v >= thr ? 1 : 0));
      const m = concept === "lamp" ? lamp : floor;
      box.innerHTML = "";
      grid("unit activation", k => MI.diverge(act[k] / sorted[0]));
      grid(`unit mask (top ${q}%)`, k => (mask[k] ? "var(--accent)" : "var(--bg-2)"));
      grid(`overlap with "${concept}"`, k => (mask[k] && m[k] ? "var(--green)" : mask[k] ? "var(--accent)" : m[k] ? "var(--blue)" : "var(--bg-2)"));
      const a = iou(mask, lamp), b = iou(mask, floor);
      $("ndOut").innerHTML = `IoU with lamp = <b>${a.toFixed(3)}</b>   ·   IoU with floor = <b>${b.toFixed(3)}</b>   →   label: <b>${a >= b ? "lamp" : "floor"}</b>` +
        `<br><span style="color:var(--green)">■</span> both  <span style="color:var(--accent)">■</span> unit only  <span style="color:var(--blue)">■</span> concept only`;
    }
    $("ndThr").oninput = draw;
    $("ndSeg").addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      concept = b.dataset.c;
      [...$("ndSeg").children].forEach(x => x.classList.toggle("on", x === b));
      draw();
    });
    draw();
    MI.onTheme(draw);
  })();

  /* ---------------- Widget: curve detector tuning ---------------- */
  (function () {
    const svg = $("tuneSvg");
    const ang = $("tuneAng"), rad = $("tuneRad");
    const D2R = Math.PI / 180;
    let spin = null;
    function resp(phi, pref, radius) {
      const c = Math.cos((phi - pref) * D2R);
      // broad tolerance for radius: the Curve Detectors article reports "a wide tolerance for the radius of curves"
      const g = Math.exp(-(Math.log(radius / 60) ** 2) / (2 * 0.9 * 0.9));
      return Math.pow(Math.max(0, c), 3) * g;
    }
    function arcPath(cx, cy, phi, R, span) {
      const u = [Math.cos(phi * D2R), -Math.sin(phi * D2R)];
      const ox = cx - u[0] * R, oy = cy - u[1] * R;
      const a0 = Math.atan2(u[1], u[0]);
      let d = "";
      for (let i = 0; i <= 30; i++) {
        const a = a0 - span + (2 * span * i) / 30;
        const x = ox + R * Math.cos(a), y = oy + R * Math.sin(a);
        d += (i ? " L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
      }
      return d;
    }
    function draw() {
      const phi = +ang.value, radius = +rad.value;
      $("tuneAngOut").textContent = phi + "°";
      $("tuneRadOut").textContent = radius + " px";
      svg.innerHTML = "";
      // stimulus
      S("rect", { x: 20, y: 20, width: 240, height: 240, rx: 12, fill: "var(--bg-2)", stroke: "var(--rule)" }, svg);
      S("text", { x: 140, y: 286, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "stimulus";
      const Rd = radius * 0.9;
      const span = Math.min(Math.PI * 0.42, 70 / Rd);
      S("path", { d: arcPath(140, 140, phi, Rd, span), fill: "none", stroke: "var(--ink)", "stroke-width": 6, "stroke-linecap": "round" }, svg);
      // detectors ring
      const cx = 480, cy = 140, RR = 92;
      S("circle", { cx, cy, r: RR, fill: "none", stroke: "var(--rule)", "stroke-dasharray": "3 4" }, svg);
      let best = 0, bi = 0;
      const rs = [];
      for (let i = 0; i < 8; i++) {
        const pref = i * 45, r = resp(phi, pref, radius);
        rs.push(r);
        if (r > best) { best = r; bi = i; }
        const x = cx + RR * Math.cos(pref * D2R), y = cy - RR * Math.sin(pref * D2R);
        S("circle", { cx: x, cy: y, r: 27, fill: MI.diverge(r), stroke: r > 0.5 ? "var(--accent)" : "var(--ink-3)", "stroke-width": r > 0.5 ? 2 : 1 }, svg);
        S("path", { d: arcPath(x, y + 0, pref, 16, 0.9), fill: "none", stroke: "var(--ink)", "stroke-width": 2.2, transform: `translate(${-Math.cos(pref * D2R) * 4},${Math.sin(pref * D2R) * 4})` }, svg);
        const lx = cx + (RR + 50) * Math.cos(pref * D2R), ly = cy - (RR + 38) * Math.sin(pref * D2R);
        S("text", { x: lx, y: ly + 4, "text-anchor": "middle", "font-size": 11.5, "font-family": "var(--mono)" }, svg).textContent = r.toFixed(2);
      }
      S("text", { x: cx, y: cy - 4, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "8 curve";
      S("text", { x: cx, y: cy + 12, "text-anchor": "middle", "font-size": 12, class: "muted" }, svg).textContent = "detectors";
      S("text", { x: 480, y: 302, "text-anchor": "middle", "font-size": 12, "font-weight": 600 }, svg).textContent =
        best > 0.05 ? `strongest: detector tuned to ${bi * 45}° (response ${best.toFixed(2)})` : "radius far from preferred size: all detectors weak";
    }
    ang.oninput = draw; rad.oninput = draw;
    $("tuneSpin").onclick = () => {
      if (spin) { clearInterval(spin); spin = null; $("tuneSpin").textContent = "▶ Rotate"; return; }
      $("tuneSpin").textContent = "■ Stop";
      spin = setInterval(() => { ang.value = (+ang.value + 3) % 360; draw(); }, 50);
    };
    draw();
    MI.onTheme(draw);
  })();
})();
