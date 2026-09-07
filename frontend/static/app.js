const state = {
  symbol: null,
  priceChart: null,
  equityChart: null,
};

async function fetchJSON(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

async function loadWatchlist() {
  const { symbols } = await fetchJSON("/api/symbols");
  const el = document.getElementById("watchlist");
  el.innerHTML = "";
  for (const sym of symbols) {
    const item = document.createElement("div");
    item.className = "watchlist-item";
    item.dataset.symbol = sym;
    item.innerHTML = `<span>${sym}</span><span class="mono">…</span>`;
    item.addEventListener("click", () => selectSymbol(sym));
    el.appendChild(item);

    fetchJSON(`/api/quote/${sym}`)
      .then((q) => {
        const chgClass = q.change_pct >= 0 ? "chg-up" : "chg-down";
        item.querySelector("span:last-child").outerHTML =
          `<span class="mono ${chgClass}">${q.change_pct >= 0 ? "+" : ""}${q.change_pct}%</span>`;
      })
      .catch(() => {});
  }
  selectSymbol(symbols[0]);
}

function setActiveWatchlistItem(symbol) {
  document.querySelectorAll(".watchlist-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.symbol === symbol);
  });
}

async function selectSymbol(symbol) {
  state.symbol = symbol;
  setActiveWatchlistItem(symbol);
  document.getElementById("symbol-title").textContent = symbol;

  await Promise.all([loadQuote(symbol), loadPriceChart(symbol), loadSignal(symbol)]);
}

async function loadQuote(symbol) {
  try {
    const q = await fetchJSON(`/api/quote/${symbol}`);
    document.getElementById("symbol-price").textContent = `$${q.price}`;
    const changeEl = document.getElementById("symbol-change");
    const up = q.change_pct >= 0;
    changeEl.textContent = `${up ? "+" : ""}${q.change} (${up ? "+" : ""}${q.change_pct}%)`;
    changeEl.style.color = up ? "#8fc389" : "#d98577";
  } catch (e) {
    document.getElementById("symbol-price").textContent = "—";
  }
}

async function loadPriceChart(symbol) {
  const { bars } = await fetchJSON(`/api/data/${symbol}?period=6mo&interval=1d`);
  const labels = bars.map((b) => b.date);
  const closes = bars.map((b) => b.close);

  const ctx = document.getElementById("priceChart").getContext("2d");
  if (state.priceChart) state.priceChart.destroy();
  state.priceChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: symbol,
        data: closes,
        borderColor: "#b08d2b",
        backgroundColor: "rgba(176,141,43,0.08)",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.15,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#a9a494", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#a9a494" }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

async function loadSignal(symbol) {
  const s = await fetchJSON(`/api/signal/${symbol}`);

  const actionTag = document.getElementById("quant-action");
  actionTag.textContent = s.action;
  actionTag.className = `action-tag ${s.action}`;
  document.getElementById("quant-score").textContent = `score ${s.score >= 0 ? "+" : ""}${s.score}`;

  const reasonsEl = document.getElementById("quant-reasons");
  reasonsEl.innerHTML = "";
  s.reasons.forEach((r) => {
    const li = document.createElement("li");
    li.textContent = r;
    reasonsEl.appendChild(li);
  });

  document.getElementById("ai-narrative").textContent = s.ai_narrative;
  document.getElementById("ai-confidence-fill").style.width = `${s.ai_confidence * 100}%`;
  document.getElementById("ai-confidence-val").textContent = s.ai_confidence.toFixed(2);
  document.getElementById("ai-source-tag").textContent = s.ai_live ? "· live" : "· offline fallback";

  const flagsEl = document.getElementById("ai-risk-flags");
  flagsEl.textContent = s.ai_risk_flags.length ? `⚑ ${s.ai_risk_flags.join(" · ")}` : "";
}

async function runBacktest() {
  const btn = document.getElementById("run-backtest");
  btn.disabled = true;
  btn.textContent = "Running…";

  try {
    const payload = {
      symbol: state.symbol,
      period: document.getElementById("bt-period").value,
      starting_cash: parseFloat(document.getElementById("bt-cash").value) || 100000,
      use_ai_review: document.getElementById("bt-ai").checked,
    };
    const result = await fetchJSON("/api/backtest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderMetrics(result.metrics);
    renderEquityChart(result.equity_curve);
    renderTradeLog(result.trades);
  } catch (e) {
    alert(`Backtest failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = "Run backtest";
  }
}

function renderMetrics(metrics) {
  const el = document.getElementById("bt-metrics");
  if (!metrics || !Object.keys(metrics).length) {
    el.innerHTML = "";
    return;
  }
  const retPos = metrics.total_return_pct >= 0;
  el.innerHTML = `
    <div class="metric"><span class="label">Total return</span><span class="value ${retPos ? "pos" : "neg"}">${retPos ? "+" : ""}${metrics.total_return_pct}%</span></div>
    <div class="metric"><span class="label">Max drawdown</span><span class="value neg">-${metrics.max_drawdown_pct}%</span></div>
    <div class="metric"><span class="label">Sharpe</span><span class="value">${metrics.sharpe_ratio}</span></div>
    <div class="metric"><span class="label">Trades</span><span class="value">${metrics.total_trades}</span></div>
    <div class="metric"><span class="label">Final equity</span><span class="value">$${metrics.final_equity.toLocaleString()}</span></div>
  `;
}

function renderEquityChart(curve) {
  const ctx = document.getElementById("equityChart").getContext("2d");
  if (state.equityChart) state.equityChart.destroy();
  state.equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: curve.map((c) => c.date),
      datasets: [{
        label: "Equity",
        data: curve.map((c) => c.equity),
        borderColor: "#3e6259",
        backgroundColor: "rgba(62,98,89,0.12)",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.1,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#a9a494", maxTicksLimit: 8 }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { ticks: { color: "#a9a494" }, grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

function renderTradeLog(trades) {
  const body = document.getElementById("trade-log-body");
  if (!trades.length) {
    body.innerHTML = `<tr><td colspan="5" class="empty-state">No trades executed for this run.</td></tr>`;
    return;
  }
  body.innerHTML = trades
    .slice()
    .reverse()
    .map((t) => `
      <tr>
        <td>${t.timestamp.slice(0, 10)}</td>
        <td class="side-${t.side}">${t.side}</td>
        <td class="mono">${t.quantity}</td>
        <td class="mono">$${t.price.toFixed(2)}</td>
        <td>${t.reasons[0] || ""}</td>
      </tr>
    `)
    .join("");
}

document.getElementById("run-backtest").addEventListener("click", runBacktest);
loadWatchlist();
