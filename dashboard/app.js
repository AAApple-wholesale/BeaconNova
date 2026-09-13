const DATA = window.BEACONNOVA_DATA || {};

const riskColorClass = {
  "蓝色": "risk-blue",
  "黄色": "risk-yellow",
  "橙色": "risk-orange",
  "红色": "risk-red",
};

const riskOrder = ["蓝色", "黄色", "橙色", "红色"];
const modelLabel = {
  "Weather-calibrated GCN": "真实天气 + 校准边权 GCN",
  "Optimized adaptive GCN": "优化版自适应 GCN",
  "Temporal TCN-GCN": "TCN-GCN 时序图模型",
  "Optimized TCN-GCN": "优化版 TCN-GCN",
};

function cnModelName(name) {
  return modelLabel[name] || name || "模型";
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return value;
  return n.toFixed(digits).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function metricFor(model, target, horizon) {
  return (model.metrics || []).find((row) => row.target === target && Number(row.horizon_min) === horizon) || {};
}

function renderModelCards() {
  const wrap = document.getElementById("modelCards");
  const temporal = (DATA.models || []).find((m) => m.name === "Optimized TCN-GCN") || (DATA.models || [])[0] || {};
  const graph = (DATA.models || []).find((m) => m.name === "Optimized adaptive GCN") || temporal;
  const person60 = metricFor(temporal, "target_person_h12", 60);
  const wait60 = metricFor(temporal, "target_wait_h12", 60);
  const graphWait60 = metricFor(graph, "target_wait_h12", 60);
  const strategy = ((DATA.strategy || {}).summary || [])[0] || {};
  const weather = DATA.weather || {};

  const cards = [
    { label: "60 分钟客流 MAE", value: fmt(person60.mae, 4), sub: `${cnModelName(temporal.name)}，测试 ${person60.rows || 0} 行` },
    { label: "60 分钟等待 WAPE", value: fmt(wait60.wape, 4), sub: `等待 MAE ${fmt(wait60.mae, 4)} 分钟；GCN 对照 ${fmt(graphWait60.wape, 4)}` },
    { label: "橙红预警减少", value: fmt(strategy.orange_red_reduction, 0), sub: `${strategy.strategy || "策略仿真"}，平均风险下降 ${fmt(strategy.avg_risk_score_delta, 4)}` },
    { label: "天气样本覆盖", value: fmt(weather.rows, 0), sub: `${weather.start || "-"} 至 ${weather.end || "-"}` },
  ];

  wrap.innerHTML = cards.map((card) => `
    <article class="metric-card">
      <p class="label">${card.label}</p>
      <p class="value">${card.value}</p>
      <p class="sub">${card.sub}</p>
    </article>
  `).join("");
}

function renderRisk(modelIndex = 0) {
  const select = document.getElementById("riskModelSelect");
  const models = DATA.models || [];
  if (!select.options.length) {
    select.innerHTML = models.map((model, idx) => `<option value="${idx}">${cnModelName(model.name)}</option>`).join("");
    select.addEventListener("change", (event) => renderRisk(Number(event.target.value)));
  }
  const model = models[modelIndex] || models[0] || {};
  const risk = (model.risk || []).filter((row) => Number(row.horizon_min) === 60 && riskOrder.includes(row.label));
  const total = risk.reduce((sum, row) => sum + Number(row.count || 0), 0) || 1;
  document.getElementById("riskBars").innerHTML = riskOrder.map((label) => {
    const item = risk.find((row) => row.label === label) || { count: 0 };
    const pct = Number(item.count || 0) / total * 100;
    return `<div class="risk-row">
      <strong>${label}</strong>
      <div class="bar-track"><div class="bar-fill ${riskColorClass[label]}" style="width:${pct}%"></div></div>
      <span>${fmt(item.count, 0)} / ${fmt(pct, 1)}%</span>
    </div>`;
  }).join("");
}

function renderHotspots() {
  const rows = (DATA.hotspots || []).slice(0, 6);
  document.getElementById("hotspotList").innerHTML = rows.map((row) => `
    <div class="hotspot-item">
      <strong>${row.node_id || "-"}</strong>
      <p>${row.datetime || "-"}，${row.risk_level || "-"}，分数 ${fmt(row.risk_score, 4)}，主因 ${row.driver || "-"}</p>
      <p>预测等待 ${fmt(row.pred_wait, 2)} 分钟，预测客流 ${fmt(row.pred_person, 2)} 人</p>
    </div>
  `).join("");
}

function renderWeather() {
  const weather = DATA.weather || {};
  const stats = [
    ["平均气温", weather.weather_temp_c, "°C"],
    ["体感温度", weather.weather_apparent_temp_c, "°C"],
    ["平均湿度", weather.weather_humidity, ""],
    ["天气惩罚", weather.weather_comfort_penalty, ""],
  ];
  document.getElementById("weatherGrid").innerHTML = stats.map(([label, data, unit]) => `
    <div class="mini-stat">
      <span>${label}</span>
      <strong>${fmt(data && data.mean, unit ? 1 : 3)}${unit}</strong>
      <small>范围 ${fmt(data && data.min, unit ? 1 : 3)} - ${fmt(data && data.max, unit ? 1 : 3)}${unit}</small>
    </div>
  `).join("");
}

function renderStrategy() {
  const summary = ((DATA.strategy || {}).summary || [])[0] || {};
  const stats = [
    ["触发干预", summary.interventions, "次"],
    ["平均舒适度提升", summary.avg_comfort_gain, "分"],
    ["平均等待压降", summary.avg_wait_reduction_min, "分钟"],
    ["橙红风险压降", summary.orange_red_reduction, "条"],
  ];
  document.getElementById("strategySummary").innerHTML = stats.map(([label, value, unit]) => `
    <div class="mini-stat">
      <span>${label}</span>
      <strong>${fmt(value, 2)}${unit}</strong>
      <small>${summary.strategy || "综合调度策略"}</small>
    </div>
  `).join("");

  const examples = ((DATA.strategy || {}).examples || []).slice(0, 6);
  document.getElementById("strategyExamples").innerHTML = examples.map((row) => `
    <tr>
      <td>${row.datetime || "-"}</td>
      <td>${row.node_id || "-"}</td>
      <td>${row.strategy || "-"}</td>
      <td>风险 -${fmt(row.risk_delta, 4)}，舒适 +${fmt(row.comfort_gain, 2)}</td>
    </tr>
  `).join("");
}

function renderEdges() {
  const rows = DATA.top_edges || [];
  document.getElementById("edgeGrid").innerHTML = rows.map((row) => `
    <div class="edge-card">
      <strong>${row.source || "-"} → ${row.target || "-"}</strong>
      <p>${row.relation || "关系"}，最佳时滞 ${row.best_lag_5min || 0} 个 5 分钟窗，相关性 ${fmt(row.lag_corr, 4)}</p>
      <div class="edge-weight"><span style="width:${Math.min(100, Number(row.calibrated_weight || 0) * 100)}%"></span></div>
    </div>
  `).join("");
}

function bindMapSwitch() {
  const mapImage = document.getElementById("mapImage");
  document.querySelectorAll("[data-map]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-map]").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      const key = button.dataset.map === "user" ? "user_map" : "official_map";
      mapImage.src = (DATA.assets || {})[key] || mapImage.src;
    });
  });
}

function init() {
  document.getElementById("generatedAt").textContent = `数据生成 ${DATA.generated_at || "-"}`;
  renderModelCards();
  renderRisk();
  renderHotspots();
  renderWeather();
  renderStrategy();
  renderEdges();
  bindMapSwitch();
}

init();

