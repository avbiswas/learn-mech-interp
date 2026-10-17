/* Chapter 9 widgets: attribution-graph explorer and two-path addition. */
(function () {
  /* ================= Attribution graph explorer ================= */
  const svg = document.getElementById("agSvg");
  if (svg) {
    const N = {
      tCap:  { x: 110, y: 360, label: '"capital"', kind: "tok" },
      tState:{ x: 250, y: 360, label: '"state"', kind: "tok" },
      tDal:  { x: 420, y: 360, label: '"Dallas"', kind: "tok" },
      tIs:   { x: 590, y: 360, label: '"is"', kind: "tok" },
      fCap:  { x: 110, y: 275, label: "capital", kind: "feat", desc: "Fires on the word 'capital'." },
      fState:{ x: 250, y: 275, label: "state", kind: "feat", desc: "Fires on 'state' (a US-state context)." },
      fDal:  { x: 420, y: 275, label: "Dallas", kind: "feat", desc: "Fires on the token 'Dallas'." },
      fSay:  { x: 180, y: 185, label: "say a capital", kind: "feat", desc: "Pushes the output toward capital-city names in general." },
      fTex:  { x: 400, y: 185, label: "Texas", kind: "feat", desc: "Represents Texas. The word never appears in the prompt: a hidden intermediate step." },
      fCal:  { x: 580, y: 185, label: "California", kind: "feat", desc: "Not active on this prompt. Used only in the swap intervention." },
      fAus:  { x: 330, y: 100, label: "say Austin", kind: "feat", desc: "Pushes the output toward 'Austin'. Needs both 'Texas' and 'say a capital'." },
      fSac:  { x: 560, y: 100, label: "say Sacramento", kind: "feat", desc: "Pushes the output toward 'Sacramento'." },
      oOther:{ x: 130, y: 25, label: "other capital", kind: "out" },
      oAus:  { x: 330, y: 25, label: "Austin", kind: "out" },
      oSac:  { x: 560, y: 25, label: "Sacramento", kind: "out" },
    };
    const E = [
      ["tCap", "fCap", 1], ["tState", "fState", 1], ["tDal", "fDal", 1],
      ["fCap", "fSay", 0.8], ["fState", "fSay", 0.2],
      ["fDal", "fTex", 0.85], ["fState", "fTex", 0.15],
      ["fTex", "fAus", 0.7], ["fSay", "fAus", 0.5], ["fDal", "fAus", 0.25],
      ["fCal", "fSac", 0.7], ["fSay", "fSac", 0.5],
      ["fAus", "oAus", 4], ["fSac", "oSac", 4], ["fSay", "oOther", 1.5],
    ];
    const order = ["tCap", "tState", "tDal", "tIs", "fCap", "fState", "fDal", "fSay", "fTex", "fCal", "fAus", "fSac", "oOther", "oAus", "oSac"];
    const thresh = { fAus: 0.5, fSac: 0.5 };
    let mode = "base";

    /* Output probabilities: rounded from the intervention figures in Lindsey et al. (2025),
       "On the Biology of a Large Language Model", Dallas case study. Only the top few outputs
       are reported there, so "<x%" means "below the smallest listed output" and the "other words" row collects the rest. Node activations below
       are a toy model and are illustrative only. */
    const PAPER = {
      base:     { "Austin": [98, "98%"], "Sacramento": [0, "<0.1%"], "other capitals": [0, "<0.1%"], "other words": [2, "~2%"] },
      suppress: { "Austin": [6, "6%"], "Sacramento": [0, "<5%"], "other capitals": [0, "<5%"], "other words": [94, "~94%"] },
      swap:     { "Austin": [0, "<0.2%"], "Sacramento": [97, "97%"], "other capitals": [0, "<0.2%"], "other words": [3, "~3%"] },
      dallas:   { "Austin": [1, "<3%"], "Sacramento": [33, "33%"], "other capitals": [37, "~37%"], "other words": [29, "~29%"] },
    };

    function compute() {
      const a = {};
      order.forEach(k => {
        if (N[k].kind === "tok") { a[k] = 1; return; }
        let s = 0;
        E.forEach(([f, t, w]) => { if (t === k) s += w * a[f]; });
        if (thresh[k]) s = Math.max(0, s - thresh[k]);
        if (N[k].kind === "feat") s = Math.min(1.2, s);
        a[k] = s;
        if (k === "fDal" && mode === "dallas") a[k] = 0;
        if (k === "fTex" && (mode === "suppress" || mode === "swap")) { a._tex = s; a[k] = 0; }
        if (k === "fTex" && mode === "dallas") a[k] = 0; // paper: Texas features fall to ~0% when Dallas is suppressed
        if (k === "fCal") a[k] = mode === "swap" ? a._tex : 0;
      });
      const p = {};
      for (const k in PAPER[mode]) p[k] = PAPER[mode][k][0] / 100;
      p["other capital"] = p["other capitals"]; // label used by the output node
      return { a, p };
    }

    let hover = null;
    function draw() {
      const { a, p } = compute();
      svg.innerHTML = "";
      const defs = MI.svg("defs", {}, svg);
      const mk = MI.svg("marker", { id: "agArr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 6, markerHeight: 6, orient: "auto" }, defs);
      MI.svg("path", { d: "M0,0 L10,5 L0,10 z", fill: "var(--accent)" }, mk);
      // layer labels
      [["tokens", 360], ["early features", 275], ["middle features", 185], ["late features", 100], ["output", 25]].forEach(([t, y]) =>
        MI.svg("text", { x: 712, y: y + 4, "text-anchor": "end", "font-size": 10, class: "muted" }, svg).textContent = t);
      // edges
      E.forEach(([f, t, w]) => {
        const s = N[f], d = N[t];
        const attr = w * a[f];
        const rel = hover && (hover === f || hover === t);
        const dim = hover && !rel;
        const y1 = s.y - 14, y2 = d.y + 16;
        const path = MI.svg("path", {
          d: `M${s.x},${y1} C ${s.x},${(y1 + y2) / 2} ${d.x},${(y1 + y2) / 2} ${d.x},${y2}`,
          fill: "none", stroke: "var(--accent)",
          "stroke-width": attr > 0.01 ? Math.min(7, 0.8 + attr * (d.kind === "out" ? 0.9 : 3.5)) : 1,
          "stroke-dasharray": attr > 0.01 ? "" : "3 4",
          opacity: dim ? 0.08 : attr > 0.01 ? 0.85 : 0.25,
          "marker-end": attr > 0.01 ? "url(#agArr)" : "",
          class: "ag-edge",
        }, svg);
        if (rel && attr > 0.01) {
          MI.svg("text", { x: (s.x + d.x) / 2 + 8, y: (y1 + y2) / 2, "font-size": 11, fill: "var(--accent)", "font-weight": 700 }, svg).textContent = attr.toFixed(2);
        }
      });
      // nodes
      order.forEach(k => {
        const n = N[k];
        const g = MI.svg("g", { class: "ag-node", tabindex: 0 }, svg);
        const w = Math.max(70, n.label.length * 7.4 + 20), h = 28;
        const act = n.kind === "out" ? (p[n.label] || 0) : a[k];
        let fill = n.kind === "tok" ? "var(--bg-2)" : MI.diverge(Math.min(1, act));
        let stroke = "var(--ink-3)", sw = 1;
        if (k === "fTex" && (mode === "suppress" || mode === "swap")) { stroke = "var(--red)"; sw = 2.5; }
        if (k === "fDal" && mode === "dallas") { stroke = "var(--red)"; sw = 2.5; }
        if (k === "fCal" && mode === "swap") { stroke = "var(--green)"; sw = 2.5; }
        if (n.kind === "out") { fill = MI.diverge(Math.min(1, act * 1.3)); }
        const dim = hover && hover !== k && !E.some(([f, t]) => (f === hover && t === k) || (t === hover && f === k));
        g.setAttribute("opacity", dim ? 0.3 : 1);
        MI.svg("rect", { x: n.x - w / 2, y: n.y - h / 2, width: w, height: h, rx: n.kind === "feat" ? 14 : 6, fill, stroke, "stroke-width": sw }, g);
        const txt = MI.svg("text", { x: n.x, y: n.y + 4, "text-anchor": "middle", "font-size": 12, "font-weight": n.kind === "out" ? 700 : 400 }, g);
        txt.textContent = n.kind === "out" ? `${n.label} ${PAPER[mode][n.label === "other capital" ? "other capitals" : n.label][1]}` : n.label;
        if (act > 0.55 && n.kind !== "tok") txt.setAttribute("fill", "#fff");
        if ((k === "fTex" && (mode === "suppress" || mode === "swap")) || (k === "fDal" && mode === "dallas")) MI.svg("text", { x: n.x, y: n.y - 20, "text-anchor": "middle", "font-size": 11, fill: "var(--red)", "font-weight": 700 }, g).textContent = "suppressed";
        if (k === "fCal" && mode === "swap") MI.svg("text", { x: n.x, y: n.y - 20, "text-anchor": "middle", "font-size": 11, fill: "var(--green)", "font-weight": 700 }, g).textContent = "injected";
        g.addEventListener("mouseenter", () => { hover = k; draw(); note(k, a); });
        g.addEventListener("focus", () => { hover = k; draw(); note(k, a); });
        g.addEventListener("mouseleave", () => { hover = null; draw(); note(null, a); });
      });
      bars();
    }
    function bars() {
      const box = document.getElementById("agBars");
      box.innerHTML = "";
      Object.entries(PAPER[mode]).forEach(([k, [v, lab]]) => {
        MI.el("div", { class: "bar-row", html: `<span>${k}</span><div class="bar-track"><div class="bar-fill ${k === "Austin" ? "" : "alt"}" style="width:${v.toFixed(1)}%"></div></div><span class="bar-val">${lab}</span>` }, box);
      });
    }
    const notes = {
      base: "Normal run: 'Dallas' activates the hidden 'Texas' step. 'Texas' and 'say a capital' together drive 'say Austin'. A weaker shortcut also goes straight from 'Dallas' to 'say Austin'. The model says 'Austin' (98%).",
      suppress: "Paper result: suppressing the 'Texas' features (to −2× their normal value) drops 'Austin' from 98% to about 6%. No other capital takes over: the top guesses become generic words like 'located' and 'not'. 'Say a capital' stays active, but without a state it has nothing specific to say.",
      swap: "Paper result: suppressing 'Texas' and injecting 'California' features (taken from an Oakland prompt) makes the model answer 'Sacramento' (97%). The answer follows the hidden intermediate state.",
      dallas: "Paper result: suppressing the 'Dallas' features removes 'Texas' and 'say Austin'. 'Say a capital' is still on, so the model names other state capitals: Albany 34%, Sacramento 33%, Springfield 3%.",
    };
    function note(k, a) {
      const el = document.getElementById("agNote");
      if (k && N[k].desc) el.textContent = `${N[k].label}: ${N[k].desc}  (illustrative activation: ${a[k].toFixed(2)})`;
      else el.textContent = notes[mode];
    }
    document.querySelectorAll("#agMode button").forEach(b => b.onclick = () => {
      mode = b.dataset.m;
      document.querySelectorAll("#agMode button").forEach(x => x.classList.toggle("on", x === b));
      draw(); note(null);
    });
    draw(); note(null);
    MI.onTheme(draw);
  }

  /* ================= Two-path addition ================= */
  const A = document.getElementById("addA");
  if (A) {
    const B = document.getElementById("addB");
    const line = document.getElementById("addLine");
    function upd() {
      const a = +A.value, b = +B.value, s = a + b;
      document.getElementById("addAo").textContent = a;
      document.getElementById("addBo").textContent = b;
      // fuzzy window of 10 consecutive numbers that contains s, not centred on it
      const lo = s - 2 - ((a + 3 * b) % 6), hi = lo + 9;
      const ra = Math.round(a / 10) * 10, rb = Math.round(b / 10) * 10;
      document.getElementById("addMag").textContent = `≈ ${lo}–${hi}`;
      document.getElementById("addMagTxt").textContent = `"about ${a}" + "about ${rb}" → "the sum is roughly ${Math.round((lo + hi) / 2)}". It cannot say which number in the range.`;
      const da = a % 10, db = b % 10, d = (da + db) % 10;
      document.getElementById("addDig").textContent = `ends in ${d}`;
      document.getElementById("addDigTxt").textContent = `"ends in ${da}" + "ends in ${db}" → lookup feature "${da}+${db}" → "ends in ${d}". It knows nothing about the size.`;
      document.getElementById("addOut").textContent = s;
      document.getElementById("addOutTxt").textContent = `The only number in ${lo}–${hi} that ends in ${d} is ${s}.`;
      // number line
      line.innerHTML = "";
      const x0 = 10, x1 = 290, span = hi - lo + 6, start = lo - 3;
      const X = v => x0 + ((v - start) / span) * (x1 - x0);
      MI.svg("line", { x1: x0, y1: 30, x2: x1, y2: 30, stroke: "var(--ink-3)" }, line);
      MI.svg("rect", { x: X(lo) - 6, y: 18, width: X(hi) - X(lo) + 12, height: 24, rx: 8, fill: "var(--blue-soft)", stroke: "var(--blue)" }, line);
      for (let v = start; v <= start + span; v++) {
        const on = v === s;
        MI.svg("line", { x1: X(v), y1: 26, x2: X(v), y2: 34, stroke: on ? "var(--accent)" : "var(--ink-3)", "stroke-width": on ? 3 : 1 }, line);
        if (v % 5 === 0 || on) MI.svg("text", { x: X(v), y: on ? 14 : 54, "text-anchor": "middle", "font-size": on ? 12 : 10, "font-weight": on ? 700 : 400, fill: on ? "var(--accent)" : "var(--ink-3)" }, line).textContent = v;
      }
    }
    A.oninput = upd; B.oninput = upd; upd();
  }
})();
