(function () {
  const article = document.querySelector('.article');
  const blocks = [...article.querySelectorAll('h1,h2,h3,h4,p,li,td,th,summary,pre,dt,dd')]
    .filter(el => !el.closest('svg,script,style,.pager'));
  blocks.forEach((el, i) => el.dataset.searchId = `search-${i}`);

  const button = document.createElement('button');
  button.id = 'searchBtn';
  button.textContent = 'Search';
  button.setAttribute('aria-haspopup', 'dialog');
  document.querySelector('#themeBtn').before(button);
  const dialog = document.createElement('dialog');
  dialog.className = 'search-dialog';
  dialog.setAttribute('aria-labelledby', 'searchLabel');
  dialog.innerHTML = `<div class="search-header"><label id="searchLabel" for="siteSearch">Search the survey</label><button class="btn" type="button" aria-label="Close search">✕</button></div>
    <input id="siteSearch" type="search" placeholder="Search a keyword, concept, or paper…" autocomplete="off">
    <p class="search-status" role="status" aria-live="polite">Search all chapters · Ctrl / ⌘ K</p>
    <div class="search-results"></div>`;
  document.body.append(dialog);
  const input = dialog.querySelector('input');
  const status = dialog.querySelector('.search-status');
  const results = dialog.querySelector('.search-results');
  function open() { dialog.showModal(); input.focus(); }
  button.onclick = open;
  dialog.querySelector('button').onclick = () => dialog.close();
  dialog.addEventListener('click', e => { if (e.target === dialog) dialog.close(); });
  document.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      if (dialog.open) dialog.close(); else open();
    }
  });
  function snippet(text, query) {
    const at = text.toLowerCase().indexOf(query.toLowerCase());
    const start = Math.max(0, at - 65);
    const end = Math.min(text.length, Math.max(start + 180, at + query.length));
    const fragment = document.createDocumentFragment();
    fragment.append((start ? '…' : '') + text.slice(start, at));
    const mark = document.createElement('mark');
    mark.textContent = text.slice(at, at + query.length);
    fragment.append(mark, text.slice(at + query.length, end) + (end < text.length ? '…' : ''));
    return fragment;
  }
  input.addEventListener('input', () => {
    const query = input.value.trim();
    results.replaceChildren();
    if (!query) { status.textContent = 'Search all chapters · Ctrl / ⌘ K'; return; }
    const matches = window.MI_SEARCH_INDEX.filter(entry => entry.text.toLowerCase().includes(query.toLowerCase()));
    status.textContent = matches.length ? `${matches.length} matching passages${matches.length > 80 ? ' · showing the first 80' : ''}` : 'No matches. Try another keyword.';
    matches.slice(0, 80).forEach(entry => {
      const link = document.createElement('a');
      link.className = 'search-result';
      link.href = `${entry.file}?q=${encodeURIComponent(query)}#${entry.id}`;
      const title = document.createElement('strong');
      title.textContent = `${entry.title} · ${entry.section}`;
      const excerpt = document.createElement('span');
      excerpt.append(snippet(entry.text, query));
      link.append(title, excerpt);
      results.append(link);
    });
  });
  input.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown' && results.firstElementChild) {
      e.preventDefault(); results.firstElementChild.focus();
    }
    if (e.key === 'Enter' && results.firstElementChild) results.firstElementChild.click();
  });

  function jump() {
    const target = blocks.find(el => `#${el.dataset.searchId}` === location.hash);
    if (!target) return;
    for (let parent = target; parent && parent !== article; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS') parent.open = true;
      if (parent.classList.contains('tab-panel')) {
        const tabs = parent.closest('.tabs');
        const panels = [...tabs.querySelectorAll(':scope > .tab-panel')];
        tabs.querySelector('.tab-buttons').children[panels.indexOf(parent)].click();
      }
      if (parent.classList.contains('step')) {
        const steps = parent.closest('.steps');
        const panels = [...steps.querySelectorAll(':scope > .step')];
        steps.querySelector('.stepper').children[panels.indexOf(parent)].click();
      }
    }
    document.querySelectorAll('.search-destination').forEach(el => el.classList.remove('search-destination'));
    target.classList.add('search-destination');
    target.setAttribute('tabindex', '-1');
    requestAnimationFrame(() => { target.focus({ preventScroll: true }); target.scrollIntoView({ block: 'center' }); });
  }
  window.addEventListener('hashchange', jump);
  window.addEventListener('load', jump);
})();
