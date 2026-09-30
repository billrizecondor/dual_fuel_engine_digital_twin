// Shared helpers for the project demos: theme toggle, chart theming, code viewer.
(function () {
  const root = document.documentElement;
  try { const t = localStorage.getItem('theme'); if (t) root.dataset.theme = t; } catch (e) {}

  const css = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const series = () => [1, 2, 3, 4, 5, 6, 7, 8].map((i) => css('--s' + i));

  // Charts are registered with a builder so they can be rebuilt when the theme changes.
  const registry = [];
  function applyDefaults() {
    if (!window.Chart) return;
    const C = window.Chart;
    C.defaults.font.family = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
    C.defaults.font.size = 12;
    C.defaults.color = css('--muted');
    C.defaults.borderColor = css('--grid');
    C.defaults.maintainAspectRatio = false;
    C.defaults.animation.duration = 300;
    C.defaults.plugins.legend.display = false; // HTML legends are used instead
    const tt = C.defaults.plugins.tooltip;
    tt.backgroundColor = css('--surface');
    tt.titleColor = css('--ink');
    tt.bodyColor = css('--ink');
    tt.borderColor = css('--line');
    tt.borderWidth = 1;
    tt.padding = 10;
    tt.boxPadding = 4;
    tt.usePointStyle = true;
    tt.cornerRadius = 8;
  }
  function chart(canvasId, build) {
    const entry = { canvasId, build, instance: null };
    registry.push(entry);
    render(entry);
    return {
      update(newBuild) { if (newBuild) entry.build = newBuild; render(entry); },
      get instance() { return entry.instance; },
    };
  }
  function render(entry) {
    applyDefaults();
    const el = document.getElementById(entry.canvasId);
    if (!el) return;
    if (entry.instance) entry.instance.destroy();
    entry.instance = new window.Chart(el, entry.build({ css, series: series() }));
  }
  function rerenderAll() { registry.forEach(render); document.dispatchEvent(new Event('themechange')); }

  function toggleTheme() {
    const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
    root.dataset.theme = dark ? 'light' : 'dark';
    try { localStorage.setItem('theme', root.dataset.theme); } catch (e) {}
    rerenderAll();
  }
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (!root.dataset.theme) rerenderAll(); });
  document.addEventListener('click', (e) => { if (e.target.closest('.theme-btn')) toggleTheme(); });

  // Minimal Python highlighter (comments, strings, keywords, numbers, function names).
  function esc(s) { return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function highlightPython(src) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:\\.|[^"\\\n])*"|f?'(?:\\.|[^'\\\n])*')|\b(def|return|import|from|as|for|in|if|elif|else|with|class|None|True|False|and|or|not|lambda|yield|try|except|raise|while|pass|continue|break|is|assert)\b|\b(\d[\d_]*\.?\d*(?:e[+-]?\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;
    let out = '', last = 0, m;
    while ((m = re.exec(src))) {
      out += esc(src.slice(last, m.index));
      const [t] = m;
      if (m[1]) out += '<span class="com">' + esc(t) + '</span>';
      else if (m[2]) out += '<span class="str">' + esc(t) + '</span>';
      else if (m[3]) out += '<span class="kw">' + t + '</span>';
      else if (m[4]) out += '<span class="num">' + t + '</span>';
      else if (m[5]) out += '<span class="fn">' + t + '</span>';
      last = m.index + t.length;
    }
    return out + esc(src.slice(last));
  }

  // Loads a file straight from the GitHub repo so the code shown always matches the source.
  async function loadCode(preId, repo, path) {
    const pre = document.getElementById(preId);
    if (!pre) return;
    const url = 'https://raw.githubusercontent.com/billrizecondor/' + repo + '/main/' + path;
    try {
      const res = await fetch(url, { cache: 'no-cache' });
      if (!res.ok) throw new Error(res.status);
      pre.innerHTML = highlightPython(await res.text());
    } catch (e) {
      pre.innerHTML = 'Could not load the code here. <a href="https://github.com/billrizecondor/' + repo + '/blob/main/' + path + '">View it on GitHub →</a>';
    }
  }

  const fmt = {
    int: (v) => Math.round(v).toLocaleString('en-US'),
    dec: (v, d = 1) => v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }),
    eurM: (v) => '€' + (v / 1e6).toLocaleString('en-US', { maximumFractionDigits: 1 }) + 'M',
    eurB: (v) => '€' + (v / 1e9).toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + 'B',
    pct: (v, d = 0) => (v * 100).toLocaleString('en-US', { maximumFractionDigits: d }) + '%',
    compact: (v) => Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(v),
  };

  // Standard axis/grid styling for bar and line charts.
  function axes({ x = {}, y = {} } = {}) {
    const grid = { color: css('--grid'), drawTicks: false };
    const border = { color: css('--line') };
    return {
      x: Object.assign({ grid: Object.assign({}, grid, { display: false }), border, ticks: { padding: 6 } }, x),
      y: Object.assign({ grid, border: { display: false }, ticks: { padding: 8 } }, y),
    };
  }

  window.Demo = { css, series, chart, rerenderAll, loadCode, fmt, axes };
})();
