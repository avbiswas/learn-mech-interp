/* ==========================================================
   Inside the Black Box — shared page behaviour
   Each page: <body data-chapter="N"> <main><article class="article">…</article></main>
   This script builds the top bar, sidebar, pager, and wires
   up components (tabs, steppers, quizzes, glossary tooltips).
   ========================================================== */
(function () {
  const CHAPTERS = [
    { n: 0,  file: "index.html",                 title: "Start here" },
    { n: 1,  file: "01-foundations.html",        title: "What is mechanistic interpretability?" },
    { n: 2,  file: "02-vision-era.html",         title: "The birth: looking inside vision models" },
    { n: 3,  file: "03-superposition.html",      title: "Features and superposition" },
    { n: 4,  file: "04-transformers.html",       title: "Transformers through a mech-interp lens" },
    { n: 5,  file: "05-causal-methods.html",     title: "Causal tools: ablation and patching" },
    { n: 6,  file: "06-circuit-case-studies.html", title: "Circuit case studies" },
    { n: 7,  file: "07-probing-steering.html",   title: "Probing and steering" },
    { n: 8,  file: "08-dictionary-learning.html", title: "Dictionary learning and SAEs" },
    { n: 9,  file: "09-circuit-tracing.html",    title: "Circuit tracing in real models" },
    { n: 10, file: "10-new-directions.html",     title: "Beyond text circuits: new directions" },
    { n: 11, file: "11-frontier.html",           title: "Uses, debates, people, and the frontier" },
    { n: 12, file: "12-reference.html",          title: "Timeline, glossary, reading list" },
  ];
  window.MI_CHAPTERS = CHAPTERS;

  const body = document.body;
  const cur = parseInt(body.dataset.chapter || "0", 10);
  const chapter = CHAPTERS.find(c => c.n === cur) || CHAPTERS[0];

  /* ---------- theme ---------- */
  try {
    const saved = localStorage.getItem("mi-theme");
    if (saved) document.documentElement.dataset.theme = saved;
  } catch (e) {}
  function toggleTheme() {
    const root = document.documentElement;
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("mi-theme", root.dataset.theme); } catch (e) {}
    document.dispatchEvent(new CustomEvent("themechange"));
  }

  /* ---------- top bar ---------- */
  const top = document.createElement("header");
  top.className = "topbar";
  top.innerHTML = `
    <button id="tocToggle" aria-label="Open contents">☰</button>
    <a class="brand" href="index.html">Inside the <span>Black Box</span></a>
    <span class="crumb">${cur > 0 ? `Ch. ${cur} · ${chapter.title}` : "An interactive survey of mechanistic interpretability"}</span>
    <span class="spacer"></span>
    <a class="btn" href="12-reference.html#glossary">Glossary</a>
    <button id="themeBtn" aria-label="Toggle dark mode">◐</button>
    <div class="progress" id="progress"></div>`;
  body.prepend(top);
  top.querySelector("#themeBtn").onclick = toggleTheme;
  top.querySelector("#tocToggle").onclick = () => body.classList.toggle("toc-open");

  /* ---------- layout + sidebar ---------- */
  const main = document.querySelector("main");
  const layout = document.createElement("div");
  layout.className = "layout";
  const side = document.createElement("nav");
  side.className = "sidebar";
  main.parentNode.insertBefore(layout, main);
  layout.appendChild(side);
  layout.appendChild(main);

  const chapterList = CHAPTERS.map(c =>
    `<li><a href="${c.file}" class="${c.n === cur ? "current" : ""}"><span class="n">${c.n === 0 ? "◦" : c.n}</span><span>${c.title}</span></a></li>`
  ).join("");

  const article = document.querySelector(".article");
  const heads = article ? [...article.querySelectorAll("h2")] : [];
  heads.forEach((h, i) => {
    if (!h.id) h.id = h.textContent.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "s" + i;
  });
  const onPage = heads.map(h => `<li><a href="#${h.id}">${h.dataset.short || h.textContent}</a></li>`).join("");

  side.innerHTML = `
    ${heads.length ? `<h4>On this page</h4><ul class="onpage">${onPage}</ul>` : ""}
    <h4>Chapters</h4><ol class="chapters">${chapterList}</ol>`;
  side.addEventListener("click", e => { if (e.target.closest("a")) body.classList.remove("toc-open"); });

  /* ---------- pager ---------- */
  if (article && !article.querySelector(".pager")) {
    const prev = CHAPTERS.find(c => c.n === cur - 1);
    const next = CHAPTERS.find(c => c.n === cur + 1);
    const pager = document.createElement("nav");
    pager.className = "pager";
    pager.innerHTML =
      (prev ? `<a class="prev" href="${prev.file}"><small>← Previous</small>${prev.title}</a>` : "") +
      (next ? `<a class="next" href="${next.file}"><small>Next →</small>${next.title}</a>` : "");
    article.appendChild(pager);
  }

  /* ---------- progress + scroll spy ---------- */
  const prog = document.getElementById("progress");
  const links = [...side.querySelectorAll(".onpage a")];
  function onScroll() {
    const h = document.documentElement;
    const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
    prog.style.width = (p * 100).toFixed(1) + "%";
    let active = 0;
    heads.forEach((hd, i) => { if (hd.getBoundingClientRect().top < 140) active = i; });
    links.forEach((a, i) => a.classList.toggle("active", i === active));
  }
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- tabs: <div class="tabs"><div class="tab-panel" data-label="A">… ---------- */
  document.querySelectorAll(".tabs").forEach(t => {
    const panels = [...t.querySelectorAll(":scope > .tab-panel")];
    const bar = document.createElement("div");
    bar.className = "tab-buttons";
    panels.forEach((p, i) => {
      const b = document.createElement("button");
      b.textContent = p.dataset.label || "Tab " + (i + 1);
      b.onclick = () => {
        panels.forEach(q => q.classList.remove("on"));
        bar.querySelectorAll("button").forEach(q => q.classList.remove("on"));
        p.classList.add("on"); b.classList.add("on");
      };
      bar.appendChild(b);
      if (i === 0) { p.classList.add("on"); b.classList.add("on"); }
    });
    t.prepend(bar);
  });

  /* ---------- steppers ----------
     <div class="steps" data-target="svgId">
        <div class="step" data-label="1. Clean run">text…</div> …
     </div>
     Elements in the target SVG with data-show="1 3" are visible only on
     those steps (1-based). The container gets data-step="k".            */
  document.querySelectorAll(".steps").forEach(s => {
    const steps = [...s.querySelectorAll(":scope > .step")];
    const bar = document.createElement("div");
    bar.className = "stepper";
    const target = s.dataset.target ? document.getElementById(s.dataset.target) : null;
    const go = k => {
      steps.forEach((st, i) => (st.style.display = i === k ? "block" : "none"));
      [...bar.children].forEach((b, i) => b.classList.toggle("on", i === k));
      s.dataset.step = k + 1;
      if (target) {
        target.dataset.step = k + 1;
        target.querySelectorAll("[data-show]").forEach(el => {
          const on = el.dataset.show.split(/\s+/).includes(String(k + 1));
          el.style.opacity = on ? 1 : 0;
          el.style.transition = "opacity .35s";
        });
      }
      s.dispatchEvent(new CustomEvent("stepchange", { detail: k + 1 }));
    };
    steps.forEach((st, i) => {
      st.classList.add("step-body");
      const b = document.createElement("button");
      b.textContent = st.dataset.label || "Step " + (i + 1);
      b.onclick = () => go(i);
      bar.appendChild(b);
    });
    const nav = document.createElement("div");
    nav.className = "controls";
    nav.innerHTML = `<button class="btn" data-d="-1">← Back</button><button class="btn primary" data-d="1">Next →</button>`;
    nav.onclick = e => {
      const d = +(e.target.dataset.d || 0);
      if (!d) return;
      const k = Math.min(steps.length - 1, Math.max(0, (+s.dataset.step || 1) - 1 + d));
      go(k);
    };
    s.prepend(bar);
    s.appendChild(nav);
    go(0);
  });

  /* ---------- quizzes ----------
     <div class="quiz" data-answer="2"><div class="q">…</div>
       <div class="opts"><button>…</button>…</div><div class="why">…</div></div> */
  document.querySelectorAll(".quiz").forEach(q => {
    if (!q.querySelector(".quiz-head")) {
      const h = document.createElement("div");
      h.className = "quiz-head"; h.textContent = "Check your understanding";
      q.prepend(h);
    }
    const ans = parseInt(q.dataset.answer, 10);
    const btns = [...q.querySelectorAll(".opts button")];
    btns.forEach((b, i) => b.onclick = () => {
      btns.forEach(x => x.classList.remove("right", "wrong"));
      b.classList.add(i === ans ? "right" : "wrong");
      if (i !== ans) btns[ans].classList.add("right");
      q.classList.add("done");
    });
  });

  /* ---------- glossary tooltips ----------
     <span class="term" data-term="superposition">superposition</span>
     or <span class="term" data-def="inline definition">word</span>       */
  const tip = document.createElement("div");
  tip.className = "tip";
  document.body.appendChild(tip);
  function showTip(el) {
    const g = window.MI_GLOSSARY || {};
    const key = (el.dataset.term || el.textContent).toLowerCase().trim();
    const def = el.dataset.def || (g[key] && g[key].def);
    if (!def) return;
    const name = (g[key] && g[key].name) || el.textContent;
    tip.innerHTML = `<b>${name}</b><br>${def}`;
    const r = el.getBoundingClientRect();
    tip.style.left = Math.max(8, Math.min(window.innerWidth - 336, r.left + window.scrollX)) + "px";
    tip.style.top = (r.bottom + window.scrollY + 8) + "px";
    tip.classList.add("show");
  }
  document.addEventListener("mouseover", e => { const t = e.target.closest(".term"); if (t) showTip(t); });
  document.addEventListener("mouseout", e => { if (e.target.closest(".term")) tip.classList.remove("show"); });
  document.addEventListener("focusin", e => { const t = e.target.closest(".term"); if (t) showTip(t); });
  document.addEventListener("focusout", () => tip.classList.remove("show"));
  document.querySelectorAll(".term").forEach(t => t.setAttribute("tabindex", "0"));

  /* ---------- small helpers for widgets ---------- */
  window.MI = {
    css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); },
    svg(tag, attrs = {}, parent) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      for (const k in attrs) el.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(el);
      return el;
    },
    el(tag, attrs = {}, parent) {
      const el = document.createElement(tag);
      for (const k in attrs) k === "text" ? (el.textContent = attrs[k]) : k === "html" ? (el.innerHTML = attrs[k]) : el.setAttribute(k, attrs[k]);
      if (parent) parent.appendChild(el);
      return el;
    },
    /* seeded RNG so demos are repeatable */
    rng(seed = 1) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); },
    gauss(r) { const u = r() || 1e-9, v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); },
    /* diverging colour: v in [-1,1] → blue..neutral..orange */
    diverge(v) {
      v = Math.max(-1, Math.min(1, v));
      const dark = document.documentElement.dataset.theme === "dark" ||
        (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      const mid = dark ? [40, 40, 46] : [247, 246, 242];
      const pos = dark ? [251, 146, 60] : [194, 65, 12];
      const neg = dark ? [122, 162, 255] : [37, 99, 235];
      const t = Math.abs(v), c = v >= 0 ? pos : neg;
      const m = mid.map((m0, i) => Math.round(m0 + (c[i] - m0) * t));
      return `rgb(${m[0]},${m[1]},${m[2]})`;
    },
    onTheme(fn) { document.addEventListener("themechange", fn); matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn); },
  };
})();
