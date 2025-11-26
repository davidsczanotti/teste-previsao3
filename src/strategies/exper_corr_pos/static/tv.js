(() => {
  const chartEl = document.getElementById("chart");
  const strategySelect = document.getElementById("strategy-select");
  const timeframeSelect = document.getElementById("timeframe-select");
  const reloadBtn = document.getElementById("reload-btn");
  const symbolLabel = document.getElementById("symbol-label");
  const tfLabel = document.getElementById("tf-label");
  const priceSell = document.getElementById("price-sell");
  const priceBuy = document.getElementById("price-buy");
  const watchlistBody = document.getElementById("watchlist-body");
  const timeButtons = document.querySelectorAll(".time-buttons button");
  const debtValue = document.getElementById("debt-value");
  const bonusValue = document.getElementById("bonus-value");
  const cashValue = document.getElementById("cash-value");
  const daysValue = document.getElementById("days-value");
  const debtBar = document.getElementById("debt-bar");
  const bonusBar = document.getElementById("bonus-bar");
  const cashBar = document.getElementById("cash-bar");
  const daysBar = document.getElementById("days-bar");
  const livesValue = document.getElementById("lives-value");
  const livesBar = document.getElementById("lives-bar");
  const moodFace = document.getElementById("mood-face");
  const moodText = document.getElementById("mood-text");
  const moodCountHappy = document.getElementById("mood-count-happy");
  const moodCountSad = document.getElementById("mood-count-sad");
  const moodCountNeutral = document.getElementById("mood-count-neutral");
  const expertsList = document.getElementById("experts-list");
  const gateTopk = document.getElementById("gate-topk");
  const gateModes = document.getElementById("gate-modes");
  const playBtn = document.getElementById("play-btn");
  const progressRange = document.getElementById("progress-range");
  let progress = 0;
  let playing = false;
  let playTimer = null;

  const chart = LightweightCharts.createChart(chartEl, {
    layout: { background: { color: "#0f1115" }, textColor: "#e7ecf3" },
    grid: {
      vertLines: { color: "rgba(255,255,255,0.04)" },
      horLines: { color: "rgba(255,255,255,0.04)" },
    },
    timeScale: { borderColor: "rgba(255,255,255,0.08)" },
    rightPriceScale: { borderColor: "rgba(255,255,255,0.08)" },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: "#6be6a4",
    downColor: "#f46f7d",
    borderUpColor: "#6be6a4",
    borderDownColor: "#f46f7d",
    wickUpColor: "#6be6a4",
    wickDownColor: "#f46f7d",
  });
  const bbUpper = chart.addLineSeries({ color: "#ff7eb6", lineWidth: 2, priceLineVisible: false });
  const bbMiddle = chart.addLineSeries({ color: "#7ab5ff", lineWidth: 2, priceLineVisible: false });
  const bbLower = chart.addLineSeries({ color: "#5de2c6", lineWidth: 2, priceLineVisible: false });
  const hmaSeries = chart.addLineSeries({ color: "#f4c95d", lineWidth: 2, priceLineVisible: false });

  function setMarkers(trades) {
    if (!trades || !trades.length) {
      candleSeries.setMarkers([]);
      return;
    }
    const markers = trades.map((t) => ({
      time: t.time,
      position: t.side === "short" ? "aboveBar" : "belowBar",
      color: t.side === "short" ? "#ff6b9f" : "#4ade80",
      shape: t.side === "short" ? "arrowDown" : "arrowUp",
      text: `${t.side === "short" ? "S" : "L"} ${t.pnl.toFixed(2)}`,
    }));
    candleSeries.setMarkers(markers);
  }

  async function loadData() {
    const strategy = strategySelect.value;
    const timeframe = timeframeSelect.value;
    const res = await fetch(`/api/tv_data?strategy=${strategy}&timeframe=${timeframe}&progress=${progress.toFixed(3)}`);
    if (!res.ok) return;
    const data = await res.json();
    symbolLabel.textContent = data.symbol || strategy.toUpperCase();
    tfLabel.textContent = data.timeframe || timeframe;
    candleSeries.setData(data.candles || []);
    bbUpper.setData(data.bb_upper || []);
    bbMiddle.setData(data.bb_middle || []);
    bbLower.setData(data.bb_lower || []);
    hmaSeries.setData(data.hma || []);
    setMarkers(data.trades || []);
    const last = (data.candles || []).at(-1);
    if (last) {
      priceSell.textContent = last.close.toFixed(2);
      priceBuy.textContent = last.close.toFixed(2);
      updateWatchlist([{ name: data.symbol || strategy, price: last.close, var: "—" }]);
    }
    updateStats(data.stats || {});
    renderExperts(data.stats || {});
    if (progressRange) {
      progressRange.value = Math.round(progress * 100);
    }
  }

  function updateWatchlist(items) {
    watchlistBody.innerHTML = "";
    items.forEach((it) => {
      const div = document.createElement("div");
      div.className = "watch-item";
      div.innerHTML = `<div class="name">${it.name}</div>
        <div class="price">${Number(it.price).toFixed(2)}</div>
        <div class="var">${it.var}</div>`;
      watchlistBody.appendChild(div);
    });
  }

  function pct(num, den) {
    if (!den || den === 0) return 0;
    return Math.max(0, Math.min(100, (num / den) * 100));
  }

  function updateStats(stats) {
    const debt = stats.debt_remaining ?? 0;
    const cash = stats.cash ?? 0;
    const bonusValueNum = stats.bonus_value ?? 0;
    const bonusPctNum = stats.bonus_pct ?? 0;
    const daysLeft = stats.days_remaining ?? 0;
    const daysTotal = stats.days_total ?? 1;
    debtValue.textContent = debt ? debt.toFixed(2) : "0.00";
    bonusValue.textContent = bonusValueNum ? `${bonusValueNum.toFixed(2)} (${(bonusPctNum*100).toFixed(0)}%)` : "0.00";
    cashValue.textContent = cash ? cash.toFixed(2) : "0.00";
    daysValue.textContent = `${daysLeft}/${daysTotal}`;
    debtBar.style.width = `${pct(debt, stats.living_cost || debt || 1)}%`;
    bonusBar.style.width = `${pct(bonusPctNum, stats.bonus_cap_pct || 1)}%`;
    cashBar.style.width = `${pct(cash, (stats.init_equity || cash || 1) * 1.5)}%`;
    daysBar.style.width = `${pct(daysLeft, daysTotal)}%`;
    const livesTotal = stats.lives_total ?? 10;
    const livesRemaining = stats.lives_remaining ?? livesTotal;
    if (livesValue) livesValue.textContent = `${livesRemaining}/${livesTotal}`;
    if (livesBar) livesBar.style.width = `${pct(livesRemaining, livesTotal)}%`;

    const mood = stats.mood || "neutral";
    if (mood === "happy") {
      moodFace.textContent = "😃";
      moodText.textContent = "Feliz";
    } else if (mood === "sad") {
      moodFace.textContent = "☹️";
      moodText.textContent = "Triste";
    } else {
      moodFace.textContent = "😐";
      moodText.textContent = "Neutro";
    }
    if (moodCountHappy && moodCountSad && moodCountNeutral) {
      moodCountHappy.textContent = stats.mood_count?.happy ?? 0;
      moodCountSad.textContent = stats.mood_count?.sad ?? 0;
      moodCountNeutral.textContent = stats.mood_count?.neutral ?? 0;
    }
  }

  function renderExperts(stats) {
    if (!expertsList) return;
    expertsList.innerHTML = "";
    const experts = stats.experts || [];
    experts.forEach((name) => {
      const div = document.createElement("div");
      div.className = "expert-item";
      div.innerHTML = `<span class="expert-dot"></span><span class="expert-name">${name}</span>`;
      expertsList.appendChild(div);
    });
    if (gateTopk) gateTopk.textContent = stats.gate_top_k ?? "--";
    if (gateModes) gateModes.textContent = stats.allow_short ? "long/short" : "long-only";
  }

  reloadBtn.addEventListener("click", loadData);
  timeButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      timeButtons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      timeframeSelect.value = btn.dataset.tf;
      loadData();
    });
  });

  // inicial
  timeframeSelect.value = "1d";
  loadData();

  function stopPlay() {
    playing = false;
    if (playTimer) {
      clearInterval(playTimer);
      playTimer = null;
    }
    if (playBtn) playBtn.textContent = "Play";
  }

  function startPlay() {
    if (playing) return;
    playing = true;
    if (playBtn) playBtn.textContent = "Pause";
    playTimer = setInterval(() => {
      progress += 0.02;
      if (progress >= 1) {
        progress = 1;
        stopPlay();
      }
      loadData();
    }, 700);
  }

  if (playBtn) {
    playBtn.addEventListener("click", () => {
      if (playing) {
        stopPlay();
      } else {
        if (progress >= 1) progress = 0;
        startPlay();
      }
    });
  }

  if (progressRange) {
    progressRange.addEventListener("input", (e) => {
      const val = Number(e.target.value || 0);
      progress = Math.max(0, Math.min(1, val / 100));
      loadData();
    });
  }

  // refresh while paused every 5s only if not playing
  setInterval(() => {
    if (!playing) loadData();
  }, 5000);
})();
