const API = "http://localhost:5000";

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
function levelClass(level) {
  if (!level) return "";
  const l = level.toLowerCase();
  if (l === "expert")       return "level-expert";
  if (l === "intermediate") return "level-intermediate";
  return "level-beginner";
}

function scorePill(val, good) {
  if (val === undefined || val === null || val === "—") return "<span class='cell-na'>—</span>";
  const v = parseFloat(val);
  let cls = "";
  if (good === "higher") cls = v >= 0.65 ? "pill-good" : v >= 0.45 ? "pill-mid" : "pill-bad";
  if (good === "lower")  cls = v <= 0.5  ? "pill-good" : v <= 1.2  ? "pill-mid" : "pill-bad";
  return `<span class="cell-score ${cls}">${v.toFixed(3)}</span>`;
}

function dbiPill(val) {
  if (val === undefined || val === null || val < 0) return "<span class='cell-na'>N/A</span>";
  const v = parseFloat(val);
  const cls = v <= 0.5 ? "pill-good" : v <= 1.2 ? "pill-mid" : "pill-bad";
  return `<span class="cell-score ${cls}">${v.toFixed(4)}</span>`;
}

function setBar(id, value) {
  const el = document.getElementById(id);
  if (el) setTimeout(() => { el.style.width = `${Math.round(Math.min(value,1)*100)}%`; }, 80);
}

function setLabel(id, label) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = label;
  el.className   = "dim-label " + label.replace(/ /g, ".");
}

function showEl(id)  { const e=document.getElementById(id); if(e) e.classList.remove("hidden"); }
function hideEl(id)  { const e=document.getElementById(id); if(e) e.classList.add("hidden"); }

// ─────────────────────────────────────────────
// Radar chart (pure Canvas)
// ─────────────────────────────────────────────
function drawRadar(scores) {
  const canvas = document.getElementById("radarChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W=canvas.width, H=canvas.height, cx=W/2, cy=H/2, R=Math.min(W,H)*0.36;
  const labels=["PD","TD","LD","BD"];
  const vals=[scores.PD,scores.TD,scores.LD,scores.BD];
  const n=labels.length;
  ctx.clearRect(0,0,W,H);

  for(let i=0;i<n;i++){
    const a=(Math.PI*2*i)/n-Math.PI/2;
    ctx.strokeStyle="#252836"; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(cx+R*Math.cos(a),cy+R*Math.sin(a)); ctx.stroke();
  }
  for(let r=1;r<=4;r++){
    ctx.beginPath(); ctx.strokeStyle="#252836"; ctx.lineWidth=1;
    for(let i=0;i<=n;i++){
      const a=(Math.PI*2*i)/n-Math.PI/2;
      const x=cx+(R*r/4)*Math.cos(a), y=cy+(R*r/4)*Math.sin(a);
      i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }
    ctx.closePath(); ctx.stroke();
  }
  ctx.beginPath();
  for(let i=0;i<n;i++){
    const a=(Math.PI*2*i)/n-Math.PI/2, rv=R*vals[i];
    const x=cx+rv*Math.cos(a), y=cy+rv*Math.sin(a);
    i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
  }
  ctx.closePath();
  ctx.fillStyle="rgba(124,106,247,0.20)"; ctx.strokeStyle="#7c6af7"; ctx.lineWidth=2;
  ctx.fill(); ctx.stroke();
  for(let i=0;i<n;i++){
    const a=(Math.PI*2*i)/n-Math.PI/2, rv=R*vals[i];
    ctx.beginPath(); ctx.arc(cx+rv*Math.cos(a),cy+rv*Math.sin(a),5,0,Math.PI*2);
    ctx.fillStyle="#7c6af7"; ctx.fill();
  }
  ctx.font="bold 13px 'Space Mono',monospace"; ctx.fillStyle="#e8eaf0"; ctx.textAlign="center";
  for(let i=0;i<n;i++){
    const a=(Math.PI*2*i)/n-Math.PI/2, lr=R+28;
    ctx.fillText(labels[i],cx+lr*Math.cos(a),cy+lr*Math.sin(a)+4);
  }
}

// ─────────────────────────────────────────────
// Load ML Metrics (Silhouette + DBI)
// ─────────────────────────────────────────────
function loadMetrics() {
  fetch(`${API}/api/metrics`)
    .then(r => { if(!r.ok) throw new Error("no metrics"); return r.json(); })
    .then(d => {
      if (!d) return;
      hideEl("metrics-loading");
      showEl("metrics-card");

      const models = ["kmeans","dbscan","agglo"];
      const names  = {"kmeans":"K-Means","dbscan":"DBSCAN","agglo":"Agglomerative"};

      let html = `
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Silhouette ↑</th>
              <th>DBI ↓</th>
              <th>Composite ↑</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>`;

      models.forEach(m => {
        const isBest = d.best_model === m;
        const sil  = d[`sil_${m}`]  ?? d[m];          // fallback old schema
        const dbi  = d[`dbi_${m}`];
        const comp = d[`comp_${m}`];
        html += `
          <tr class="${isBest ? 'row-best' : ''}">
            <td><strong>${names[m]}</strong>${isBest ? ' <span class="crown">🏆</span>':''}</td>
            <td>${scorePill(sil, "higher")}</td>
            <td>${dbiPill(dbi)}</td>
            <td>${scorePill(comp, "higher")}</td>
            <td>${isBest ? '<span class="badge yes">Selected</span>' : '<span class="badge neutral">—</span>'}</td>
          </tr>`;
      });

      html += `</tbody></table>
        <div class="metrics-note">
          <span>Silhouette: higher=better (−1→1)</span>
          <span>DBI: lower=better (0=ideal)</span>
          <span>Composite = Silhouette + (1 − DBI_norm)</span>
        </div>`;

      document.getElementById("metrics-card").innerHTML = html;
    })
    .catch(() => {
      const el = document.getElementById("metrics-loading");
      if(el) el.textContent = "No metrics yet — run ML first.";
    });
}

// ─────────────────────────────────────────────
// Load Game Evaluation
// ─────────────────────────────────────────────
function loadGameEval() {
  hideEl("eval-card");
  const loadingEl = document.getElementById("eval-loading");
  loadingEl.textContent = "Loading…"; showEl("eval-loading");

  fetch(`${API}/api/game-evaluation`)
    .then(r => { if(!r.ok) throw new Error("No evaluation yet"); return r.json(); })
    .then(d => {
      hideEl("eval-loading"); showEl("eval-card");

      document.getElementById("eval-rating").textContent  = d.verdict?.rating  || d.rating  || "—";
      document.getElementById("eval-summary").textContent = d.verdict?.summary  || d.summary || "";

      const sat = d.verdict?.satisfied ?? d.player_satisfied;
      const lrn = d.verdict?.learning  ?? d.learning_achieved;

      const bs = document.getElementById("badge-satisfied");
      bs.textContent = sat ? "✓ Players Satisfied" : "✗ Players Not Satisfied";
      bs.className   = "badge " + (sat ? "yes" : "no");

      const bl = document.getElementById("badge-learning");
      bl.textContent = lrn ? "✓ Learning Achieved" : "✗ Learning Not Achieved";
      bl.className   = "badge " + (lrn ? "yes" : "no");

      const scores = d.avg_scores || {};
      const labels = d.dim_labels || {};
      const crit   = d.avg_criteria || {};
      const recs   = d.suggestions  || [];

      const dims = [
        {key:"PD",icon:"📚",name:"Pedagogical"},
        {key:"TD",icon:"⚙️",name:"Technological"},
        {key:"LD",icon:"🎮",name:"Ludic"},
        {key:"BD",icon:"🧠",name:"Behavioural"},
      ];

      dims.forEach(dim => {
        const k  = dim.key.toLowerCase();
        const sc = scores[dim.key] ?? 0;
        const lb = labels[dim.key] ?? "—";
        const rec = recs.find(r => r.dimension===dim.key)?.suggestion || "";
        setLabel(`${k}-label`, lb);
        setBar(`${k}-bar`, sc);
        document.getElementById(`${k}-score`).textContent = sc.toFixed ? sc.toFixed(3) : sc;
        document.getElementById(`${k}-rec`).textContent   = rec;

        // sub-criteria mini table
        const critMap = crit[dim.key] || {};
        const critEl  = document.getElementById(`${k}-crit`);
        if (critEl && Object.keys(critMap).length) {
          critEl.innerHTML = Object.entries(critMap).map(([c,v]) =>
            `<span class="crit-chip">${c}: <strong>${v.toFixed(3)}</strong></span>`
          ).join("");
        }
      });

      drawRadar({
        PD: scores.PD||0, TD: scores.TD||0, LD: scores.LD||0, BD: scores.BD||0
      });

      // Per player-type
      const prof = d.by_player_type || {};
      const ptEl = document.getElementById("player-type-grid");
      if (ptEl) {
        ptEl.innerHTML = ["beginner","intermediate","expert"].map(lv => {
          const p = prof[lv] || {};
          if (!p.count) return `<div class="pt-card pt-${lv}"><div class="pt-name">${lv}</div><div class="pt-none">No data</div></div>`;
          return `
            <div class="pt-card pt-${lv}">
              <div class="pt-name">${lv} <span class="pt-count">n=${p.count}</span></div>
              <div class="pt-row"><span>Global</span><strong>${(p.avg_global||0).toFixed(3)}</strong></div>
              <div class="pt-row"><span>PD</span><strong>${(p.avg_PD||0).toFixed(3)}</strong></div>
              <div class="pt-row"><span>TD</span><strong>${(p.avg_TD||0).toFixed(3)}</strong></div>
              <div class="pt-row"><span>LD</span><strong>${(p.avg_LD||0).toFixed(3)}</strong></div>
              <div class="pt-row"><span>BD</span><strong>${(p.avg_BD||0).toFixed(3)}</strong></div>
            </div>`;
        }).join("");
      }
    })
    .catch(err => {
      const el = document.getElementById("eval-loading");
      if(el) { el.textContent = "⚠ " + err.message; showEl("eval-loading"); }
    });
}

// ─────────────────────────────────────────────
// Load Fuzzy Player Table — FIXED
// Always fetches fresh data and shows ALL players
// sorted by player_id ascending
// ─────────────────────────────────────────────
function loadFuzzy() {
  const loadEl = document.getElementById("fuzzy-loading");
  const wrapEl = document.getElementById("fuzzy-table-wrap");
  const bodyEl = document.getElementById("fuzzy-body");
  const countEl= document.getElementById("fuzzy-count");

  loadEl.textContent = "Loading latest player evaluations…";
  showEl("fuzzy-loading"); hideEl("fuzzy-table-wrap");

  // Force cache-bust so browser never serves a stale response
  fetch(`${API}/api/fuzzy-results?t=${Date.now()}`)
    .then(r => { if(!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      hideEl("fuzzy-loading");

      if (!data || data.length === 0) {
        loadEl.textContent = "⏳ No fuzzy results yet — waiting for ML pipeline…";
        showEl("fuzzy-loading");
        return;
      }

      // Sort by player_id so newest IDs appear at bottom, or reverse for newest-first
      data.sort((a, b) => b.player_id - a.player_id);   // newest first

      if (countEl) countEl.textContent = `${data.length} players`;
      showEl("fuzzy-table-wrap");

      bodyEl.innerHTML = data.map(p => {
        const pdLabel = p.PD_label || "";
        const tdLabel = p.TD_label || "";
        const ldLabel = p.LD_label || "";
        const bdLabel = p.BD_label || "";
        return `
          <tr>
            <td><strong>#${p.player_id}</strong></td>
            <td>
              ${scorePill(p.PD, "higher")}
              <span class="sub-label">${pdLabel}</span>
            </td>
            <td>
              ${scorePill(p.TD, "higher")}
              <span class="sub-label">${tdLabel}</span>
            </td>
            <td>
              ${scorePill(p.LD, "higher")}
              <span class="sub-label">${ldLabel}</span>
            </td>
            <td>
              ${scorePill(p.BD, "higher")}
              <span class="sub-label">${bdLabel}</span>
            </td>
            <td>${scorePill(p.fuzzy_score || p.global_score, "higher")}</td>
            <td class="${levelClass(p.ml_level || p.ML_level)}">${p.ml_level || p.ML_level || "—"}</td>
          </tr>`;
      }).join("");
    })
    .catch(err => {
      loadEl.textContent = "❌ Error: " + err.message;
      showEl("fuzzy-loading");
    });
}

// ─────────────────────────────────────────────
// Load ML Table
// ─────────────────────────────────────────────
function loadML() {
  const loadEl = document.getElementById("ml-loading");
  const wrapEl = document.getElementById("ml-table-wrap");
  const bodyEl = document.getElementById("ml-body");

  loadEl.textContent = "Loading…"; showEl("ml-loading"); hideEl("ml-table-wrap");

  fetch(`${API}/api/ml-results?t=${Date.now()}`)
    .then(r => r.json())
    .then(data => {
      hideEl("ml-loading");
      if (!data.length) { loadEl.textContent = "No ML data yet."; showEl("ml-loading"); return; }

      data.sort((a,b) => b.player_id - a.player_id);
      showEl("ml-table-wrap");
      bodyEl.innerHTML = data.map(p => `
        <tr>
          <td><strong>#${p.player_id}</strong></td>
          <td class="${levelClass(p.level)}">${p.level||"—"}</td>
          <td>${p.cluster !== undefined ? p.cluster : "—"}</td>
          <td><code>${p.model||"—"}</code></td>
          <td>${scorePill(p.score, "higher")}</td>
        </tr>`).join("");
    })
    .catch(err => { loadEl.textContent="Error: "+err.message; showEl("ml-loading"); });
}

// ─────────────────────────────────────────────
// Auto-refresh every 15 s (matches ML pipeline cadence)
// ─────────────────────────────────────────────
let autoRefreshTimer = null;

function startAutoRefresh() {
  stopAutoRefresh();
  autoRefreshTimer = setInterval(() => {
    loadFuzzy();
    loadML();
    loadMetrics();
  }, 15000);
  document.getElementById("auto-badge").textContent = "Auto-refresh ON";
}

function stopAutoRefresh() {
  if (autoRefreshTimer) { clearInterval(autoRefreshTimer); autoRefreshTimer=null; }
  const b = document.getElementById("auto-badge");
  if(b) b.textContent = "Auto-refresh OFF";
}

function toggleAutoRefresh() {
  if (autoRefreshTimer) stopAutoRefresh();
  else startAutoRefresh();
}

// ─────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  loadMetrics();
  loadFuzzy();
  loadML();
  loadGameEval();
  startAutoRefresh();
});
