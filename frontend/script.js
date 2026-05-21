/**
 * script.js — SeriousEval Dashboard
 * Méthode d'intégration : Average Score Fusion (ASF)
 *
 * Données ASF clés par joueur (fuzzy_results) :
 *   fuzzy_score / global_score  → Γ(xp) = Σ_D w̄_D · Φ_D(xp)   [Eq.1]
 *   PD / TD / LD / BD          → scores de dimension Φ_D(xp)
 *   ml_level                   → label GMM  (annotatif, non-influent)
 *   decision                   → seuil Fuzzy AHP (beginner/intermediate/expert)
 *   integration_method         → "ASF"
 *
 * Données de groupe (game_evaluation.by_player_type) :
 *   avg_global  → Γ*_L = (1/|G_L|) Σ Γ(xp)          [Eq.2]
 *   avg_PD/TD/LD/BD → Φ̄_L_D                          [Eq.2]
 */

const API = "http://localhost:5000";

// ─────────────────────────────────────────────────────────────────────────────
// HELPERS GÉNÉRAUX
// ─────────────────────────────────────────────────────────────────────────────

/** Classe CSS pour un niveau de joueur (GMM ou décision Fuzzy) */
function levelClass(level) {
  if (!level) return "";
  const l = level.toLowerCase();
  if (l === "expert")       return "level-expert";
  if (l === "intermediate") return "level-intermediate";
  return "level-beginner";
}

/** Pilule colorée pour un score [0-1] */
function scorePill(val, good = "higher") {
  if (val === undefined || val === null || val === "—")
    return "<span class='cell-na'>—</span>";
  const v = parseFloat(val);
  let cls = "";
  if (good === "higher") cls = v >= 0.65 ? "pill-good" : v >= 0.45 ? "pill-mid" : "pill-bad";
  if (good === "lower")  cls = v <= 0.5  ? "pill-good" : v <= 1.2  ? "pill-mid" : "pill-bad";
  return `<span class="cell-score ${cls}">${v.toFixed(3)}</span>`;
}

function dbiPill(val) {
  if (val === undefined || val === null || val < 0)
    return "<span class='cell-na'>N/A</span>";
  const v = parseFloat(val);
  const cls = v <= 0.5 ? "pill-good" : v <= 1.2 ? "pill-mid" : "pill-bad";
  return `<span class="cell-score ${cls}">${v.toFixed(4)}</span>`;
}

/** Étiquette qualitative → classe CSS */
function qualLabel(lbl) {
  if (!lbl) return "";
  const map = {
    "Excellent": "lbl-excellent",
    "Good":      "lbl-good",
    "Fair":      "lbl-fair",
    "Poor":      "lbl-poor",
    "Very Poor": "lbl-verypoor",
  };
  return map[lbl] || "";
}

/** Badge GMM — visuel distinct pour montrer que c'est annotatif */
function gmmBadge(level) {
  if (!level) return "<span class='cell-na'>—</span>";
  const cls = levelClass(level);
  return `<span class="gmm-badge ${cls}" title="Label GMM — annotatif uniquement">${level}</span>`;
}

/** Badge décision Fuzzy AHP */
function fuzzyDecisionBadge(decision) {
  if (!decision) return "<span class='cell-na'>—</span>";
  const cls = levelClass(decision);
  return `<span class="fuzzy-decision ${cls}" title="Décision Fuzzy AHP (seuil sur Γ)">${decision}</span>`;
}

function setBar(id, value) {
  const el = document.getElementById(id);
  if (el) setTimeout(() => { el.style.width = `${Math.round(Math.min(value, 1) * 100)}%`; }, 80);
}

function setLabel(id, lbl) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = lbl;
  el.className   = "dim-label " + lbl.replace(/ /g, ".");
}

function showEl(id) { const e = document.getElementById(id); if (e) e.classList.remove("hidden"); }
function hideEl(id) { const e = document.getElementById(id); if (e) e.classList.add("hidden"); }

// ─────────────────────────────────────────────────────────────────────────────
// BANNIÈRE ASF — affichée une fois dans la page
// ─────────────────────────────────────────────────────────────────────────────

function renderASFBanner() {
  const el = document.getElementById("asf-banner");
  if (!el) return;
  el.innerHTML = `
    <div class="asf-banner">
      <span class="asf-icon">∑</span>
      <div class="asf-text">
        <strong>Integration Method: Average Score Fusion (ASF)</strong>
        <span>Γ(x<sub>p</sub>) = Σ<sub>D</sub> w̄<sub>D</sub> · Φ<sub>D</sub>(x<sub>p</sub>)</span>
      </div>
      <div class="asf-note">
        GMM label (Beginner / Intermediate / Expert) is <em>annotative only</em> — 
        it does not influence the numerical Fuzzy AHP score.
        Group statistics Γ*<sub>L</sub> are computed post-hoc for analysis.
      </div>
    </div>`;
}

// ─────────────────────────────────────────────────────────────────────────────
// RADAR CHART (Canvas pur)
// ─────────────────────────────────────────────────────────────────────────────

function drawRadar(scores) {
  const canvas = document.getElementById("radarChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height, cx = W / 2, cy = H / 2;
  const R = Math.min(W, H) * 0.36;
  const labels = ["PD", "TD", "LD", "BD"];
  const vals   = [scores.PD, scores.TD, scores.LD, scores.BD];
  const n = labels.length;
  ctx.clearRect(0, 0, W, H);

  // Axes + grille
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    ctx.strokeStyle = "#252836"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(cx, cy);
    ctx.lineTo(cx + R * Math.cos(a), cy + R * Math.sin(a));
    ctx.stroke();
  }
  for (let r = 1; r <= 4; r++) {
    ctx.beginPath(); ctx.strokeStyle = "#252836"; ctx.lineWidth = 1;
    for (let i = 0; i <= n; i++) {
      const a = (Math.PI * 2 * i) / n - Math.PI / 2;
      const x = cx + (R * r / 4) * Math.cos(a), y = cy + (R * r / 4) * Math.sin(a);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath(); ctx.stroke();
  }

  // Polygone des scores
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const rv = R * vals[i];
    const x = cx + rv * Math.cos(a), y = cy + rv * Math.sin(a);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle   = "rgba(124,106,247,0.20)";
  ctx.strokeStyle = "#7c6af7"; ctx.lineWidth = 2;
  ctx.fill(); ctx.stroke();

  // Points
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const rv = R * vals[i];
    ctx.beginPath();
    ctx.arc(cx + rv * Math.cos(a), cy + rv * Math.sin(a), 5, 0, Math.PI * 2);
    ctx.fillStyle = "#7c6af7"; ctx.fill();
  }

  // Labels
  ctx.font = "bold 13px 'Space Mono',monospace";
  ctx.fillStyle = "#e8eaf0"; ctx.textAlign = "center";
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const lr = R + 28;
    ctx.fillText(labels[i], cx + lr * Math.cos(a), cy + lr * Math.sin(a) + 4);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// ML METRICS
// ─────────────────────────────────────────────────────────────────────────────

function loadMetrics() {
  fetch(`${API}/api/metrics`)
    .then(r => { if (!r.ok) throw new Error("no metrics"); return r.json(); })
    .then(d => {
      if (!d) return;
      hideEl("metrics-loading"); showEl("metrics-card");

      const models = ["kmeans", "dbscan", "agglo"];
      const names  = { kmeans: "K-Means", dbscan: "DBSCAN", agglo: "Agglomerative" };

      let html = `
        <table class="metrics-table">
          <thead><tr>
            <th>Model</th><th>Silhouette ↑</th><th>DBI ↓</th>
            <th>Composite ↑</th><th>Status</th>
          </tr></thead><tbody>`;

      models.forEach(m => {
        const isBest = d.best_model === m;
        const sil  = d[`sil_${m}`]  ?? d[m];
        const dbi  = d[`dbi_${m}`];
        const comp = d[`comp_${m}`];
        html += `
          <tr class="${isBest ? "row-best" : ""}">
            <td><strong>${names[m]}</strong>${isBest ? ' <span class="crown">🏆</span>' : ""}</td>
            <td>${scorePill(sil, "higher")}</td>
            <td>${dbiPill(dbi)}</td>
            <td>${scorePill(comp, "higher")}</td>
            <td>${isBest
              ? '<span class="badge yes">Selected</span>'
              : '<span class="badge neutral">—</span>'}</td>
          </tr>`;
      });

      html += `</tbody></table>
        <div class="metrics-note">
          <span>Silhouette: higher=better (−1→1)</span>
          <span>DBI: lower=better (0=ideal)</span>
          <span>Composite = Silhouette + (1 − DBI_norm)</span>
          <span class="asf-note-inline">GMM label used as annotation in ASF pipeline</span>
        </div>`;

      document.getElementById("metrics-card").innerHTML = html;
    })
    .catch(() => {
      const el = document.getElementById("metrics-loading");
      if (el) el.textContent = "No metrics yet — run ML first.";
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// GAME EVALUATION (ASF)
// ─────────────────────────────────────────────────────────────────────────────

function loadGameEval() {
  hideEl("eval-card");
  const loadingEl = document.getElementById("eval-loading");
  loadingEl.textContent = "Loading…"; showEl("eval-loading");

  fetch(`${API}/api/game-evaluation`)
    .then(r => { if (!r.ok) throw new Error("No evaluation yet"); return r.json(); })
    .then(d => {
      hideEl("eval-loading"); showEl("eval-card");

      // ── Verdict global ──────────────────────────────────────────────────
      document.getElementById("eval-rating").textContent  = d.verdict?.rating  || "—";
      document.getElementById("eval-summary").textContent = d.verdict?.summary  || "";

      const sat = d.verdict?.satisfied ?? false;
      const lrn = d.verdict?.learning  ?? false;
      const bs = document.getElementById("badge-satisfied");
      bs.textContent = sat ? "✓ Players Satisfied" : "✗ Players Not Satisfied";
      bs.className   = "badge " + (sat ? "yes" : "no");
      const bl = document.getElementById("badge-learning");
      bl.textContent = lrn ? "✓ Learning Achieved" : "✗ Learning Not Achieved";
      bl.className   = "badge " + (lrn ? "yes" : "no");

      // ── Badge méthode ASF ───────────────────────────────────────────────
      const bm = document.getElementById("badge-method");
      if (bm) {
        bm.textContent = "ASF Integration";
        bm.className   = "badge neutral";
        bm.title       = "Average Score Fusion — GMM label annotatif uniquement";
      }

      // ── Scores de dimension Φ_D (scores bruts, sans PPIF) ───────────────
      const scores = d.avg_scores || {};
      const labels = d.dim_labels || {};
      const crit   = d.avg_criteria || {};
      const recs   = d.suggestions  || [];

      const dims = [
        { key: "PD", k: "pd", icon: "📚", name: "Pedagogical" },
        { key: "TD", k: "td", icon: "⚙️",  name: "Technological" },
        { key: "LD", k: "ld", icon: "🎮", name: "Ludic" },
        { key: "BD", k: "bd", icon: "🧠", name: "Behavioural" },
      ];

      dims.forEach(dim => {
        const sc  = scores[dim.key] ?? 0;
        const lb  = labels[dim.key] ?? "—";
        const rec = recs.find(r => r.dimension === dim.key)?.suggestion || "";
        setLabel(`${dim.k}-label`, lb);
        setBar(`${dim.k}-bar`, sc);
        document.getElementById(`${dim.k}-score`).textContent =
          sc.toFixed ? sc.toFixed(3) : sc;
        document.getElementById(`${dim.k}-rec`).textContent = rec;

        const critMap = crit[dim.key] || {};
        const critEl  = document.getElementById(`${dim.k}-crit`);
        if (critEl && Object.keys(critMap).length) {
          critEl.innerHTML = Object.entries(critMap).map(([c, v]) =>
            `<span class="crit-chip">${c}: <strong>${v.toFixed(3)}</strong></span>`
          ).join("");
        }
      });

      // ── Radar (scores de dimension bruts) ───────────────────────────────
      drawRadar({ PD: scores.PD || 0, TD: scores.TD || 0,
                  LD: scores.LD || 0, BD: scores.BD || 0 });

      // ── Statistiques de groupe ASF — Γ*_L et Φ̄_L_D ────────────────────
      renderGroupStats(d.by_player_type || {}, d.integration || {});
    })
    .catch(err => {
      const el = document.getElementById("eval-loading");
      if (el) { el.textContent = "⚠ " + err.message; showEl("eval-loading"); }
    });
}

/**
 * Affiche les statistiques de groupe ASF.
 * Γ*_L  = avg_global  (moyenne du score Fuzzy sur le groupe GMM)
 * Φ̄_L_D = avg_PD/TD/LD/BD
 */
function renderGroupStats(prof, integration) {
  const ptEl = document.getElementById("player-type-grid");
  if (!ptEl) return;

  const levels = ["beginner", "intermediate", "expert"];
  ptEl.innerHTML = levels.map(lv => {
    const p = prof[lv] || {};
    if (!p.count) return `
      <div class="pt-card pt-${lv}">
        <div class="pt-name">${lv}</div>
        <div class="pt-none">No players in this GMM group</div>
      </div>`;

    return `
      <div class="pt-card pt-${lv}">
        <div class="pt-name">
          ${lv}
          <span class="pt-count">n=${p.count}</span>
          <span class="pt-gmm-tag" title="Groupe défini par le GMM">GMM</span>
        </div>
        <!-- Γ*_L — score moyen Fuzzy AHP du groupe -->
        <div class="pt-row pt-row-gamma">
          <span>Γ*<sub>${lv[0].toUpperCase()}</sub></span>
          <strong>${(p.avg_global || 0).toFixed(3)}</strong>
        </div>
        <!-- Φ̄_L_D — score moyen par dimension -->
        <div class="pt-divider"></div>
        ${["PD","TD","LD","BD"].map(d => `
          <div class="pt-row">
            <span>Φ̄ ${d}</span>
            <strong>${(p[`avg_${d}`] || 0).toFixed(3)}</strong>
          </div>`).join("")}
      </div>`;
  }).join("");

  // Note sur la méthode d'intégration
  const noteEl = document.getElementById("asf-integration-note");
  if (noteEl && integration.method) {
    noteEl.innerHTML = `
      <div class="integration-note">
        <span class="int-icon">∑</span>
        <div>
          <strong>${integration.method}</strong>
          <br><small>${integration.description || ""}</small>
        </div>
      </div>`;
    showEl("asf-integration-note");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TABLE FUZZY RESULTS (ASF)
//
// Colonnes :
//   Player | Φ_PD | Φ_TD | Φ_LD | Φ_BD | Γ(xp) | GMM Label | Fuzzy Decision
// ─────────────────────────────────────────────────────────────────────────────

function loadFuzzy() {
  const loadEl  = document.getElementById("fuzzy-loading");
  const wrapEl  = document.getElementById("fuzzy-table-wrap");
  const bodyEl  = document.getElementById("fuzzy-body");
  const countEl = document.getElementById("fuzzy-count");

  loadEl.textContent = "Loading latest ASF evaluations…";
  showEl("fuzzy-loading"); hideEl("fuzzy-table-wrap");

  fetch(`${API}/api/fuzzy-results?t=${Date.now()}`)
    .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
    .then(data => {
      hideEl("fuzzy-loading");

      if (!data || data.length === 0) {
        loadEl.textContent = "⏳ No ASF results yet — waiting for ML pipeline…";
        showEl("fuzzy-loading");
        return;
      }

      data.sort((a, b) => b.player_id - a.player_id);
      if (countEl) countEl.textContent = `${data.length} players`;
      showEl("fuzzy-table-wrap");

      bodyEl.innerHTML = data.map(p => {
        // Support deux structures : fuzzy_ahp.py (PD/TD/LD/BD direct)
        // et fuzzy_evaluation.py (dimension_scores.PD etc.)
        const PD = p.PD ?? p.dimension_scores?.PD;
        const TD = p.TD ?? p.dimension_scores?.TD;
        const LD = p.LD ?? p.dimension_scores?.LD;
        const BD = p.BD ?? p.dimension_scores?.BD;
        const gamma   = p.fuzzy_score ?? p.global_score;
        const mlLevel = p.ml_level   || "—";
        const decision = p.decision || (gamma === undefined ? "—" : gamma >= 0.70 ? "expert" : gamma >= 0.45 ? "intermediate" : "beginner");
        const method  = p.integration_method || "";

        return `
          <tr>
            <td>
              <strong>#${p.player_id}</strong>
              ${method === "ASF" ? '<span class="asf-tag">ASF</span>' : ""}
            </td>
            <td>
              ${scorePill(PD, "higher")}
              <span class="sub-label">${p.PD_label || ""}</span>
            </td>
            <td>
              ${scorePill(TD, "higher")}
              <span class="sub-label">${p.TD_label || ""}</span>
            </td>
            <td>
              ${scorePill(LD, "higher")}
              <span class="sub-label">${p.LD_label || ""}</span>
            </td>
            <td>
              ${scorePill(BD, "higher")}
              <span class="sub-label">${p.BD_label || ""}</span>
            </td>
            <td>
              <!-- Γ(xp) — score ASF final -->
              ${scorePill(gamma, "higher")}
            </td>
            <td>
              <!-- GMM label : annotatif uniquement -->
              ${gmmBadge(mlLevel)}
            </td>
            <td>
              <!-- Décision Fuzzy AHP basée sur seuils de Γ -->
              ${fuzzyDecisionBadge(decision)}
            </td>
          </tr>`;
      }).join("");
    })
    .catch(err => {
      loadEl.textContent = "❌ Error: " + err.message;
      showEl("fuzzy-loading");
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// TABLE ML RESULTS (GMM classification)
// ─────────────────────────────────────────────────────────────────────────────

function loadML() {
  const loadEl = document.getElementById("ml-loading");
  const bodyEl = document.getElementById("ml-body");

  loadEl.textContent = "Loading…"; showEl("ml-loading");
  hideEl("ml-table-wrap");

  fetch(`${API}/api/ml-results?t=${Date.now()}`)
    .then(r => r.json())
    .then(data => {
      hideEl("ml-loading");
      if (!data.length) {
        loadEl.textContent = "No ML data yet.";
        showEl("ml-loading");
        return;
      }
      data.sort((a, b) => b.player_id - a.player_id);
      showEl("ml-table-wrap");
      bodyEl.innerHTML = data.map(p => `
        <tr>
          <td><strong>#${p.player_id}</strong></td>
          <td class="${levelClass(p.level)}">${p.level || "—"}</td>
          <td>${p.cluster !== undefined ? p.cluster : "—"}</td>
          <td><code>${p.model || "—"}</code></td>
          <td>${scorePill(p.score, "higher")}</td>
          <td>
            <span class="gmm-role-badge" title="Ce label est utilisé comme annotation dans le pipeline ASF">
              annotative in ASF
            </span>
          </td>
        </tr>`).join("");
    })
    .catch(err => {
      loadEl.textContent = "Error: " + err.message;
      showEl("ml-loading");
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// AUTO-REFRESH
// ─────────────────────────────────────────────────────────────────────────────

let autoRefreshTimer = null;

function startAutoRefresh() {
  stopAutoRefresh();
  autoRefreshTimer = setInterval(() => {
    loadFuzzy();
    loadML();
    loadMetrics();
    loadGameEval();
  }, 15000);
  document.getElementById("auto-badge").textContent = "Auto-refresh ON";
}

function stopAutoRefresh() {
  if (autoRefreshTimer) { clearInterval(autoRefreshTimer); autoRefreshTimer = null; }
  const b = document.getElementById("auto-badge");
  if (b) b.textContent = "Auto-refresh OFF";
}

function toggleAutoRefresh() {
  if (autoRefreshTimer) stopAutoRefresh();
  else startAutoRefresh();
}

// ─────────────────────────────────────────────────────────────────────────────
// INIT
// ─────────────────────────────────────────────────────────────────────────────

window.addEventListener("DOMContentLoaded", () => {
  renderASFBanner();
  loadMetrics();
  loadFuzzy();
  loadML();
  loadGameEval();
  startAutoRefresh();
});