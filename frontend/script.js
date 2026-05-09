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

function scorePill(val) {
  if (val === undefined || val === null || val === "—") return "—";
  return `<span class="cell-score">${parseFloat(val).toFixed(3)}</span>`;
}

function setBar(id, value) {
  const el = document.getElementById(id);
  if (el) setTimeout(() => el.style.width = `${Math.round(value * 100)}%`, 80);
}

function setLabel(id, label) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = label;
  // apply colour class based on label word
  el.className = "dim-label " + label.replace(/ /g, ".");
}

// ─────────────────────────────────────────────
// Radar chart (pure Canvas — no external libs)
// ─────────────────────────────────────────────
function drawRadar(scores) {
  const canvas = document.getElementById("radarChart");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W = canvas.width, H = canvas.height;
  const cx = W / 2, cy = H / 2;
  const R  = Math.min(W, H) * 0.36;

  const labels = ["PD", "TD", "LD", "BD"];
  const vals   = [scores.PD, scores.TD, scores.LD, scores.BD];
  const n      = labels.length;

  ctx.clearRect(0, 0, W, H);

  // axes
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const x = cx + R * Math.cos(a);
    const y = cy + R * Math.sin(a);
    ctx.strokeStyle = "#252836";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  // rings
  for (let r = 1; r <= 4; r++) {
    ctx.beginPath();
    ctx.strokeStyle = "#252836";
    ctx.lineWidth   = 1;
    for (let i = 0; i <= n; i++) {
      const a = (Math.PI * 2 * i) / n - Math.PI / 2;
      const rx = cx + (R * r / 4) * Math.cos(a);
      const ry = cy + (R * r / 4) * Math.sin(a);
      if (i === 0) ctx.moveTo(rx, ry); else ctx.lineTo(rx, ry);
    }
    ctx.closePath();
    ctx.stroke();
  }

  // data polygon
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const rv = R * vals[i];
    const x = cx + rv * Math.cos(a);
    const y = cy + rv * Math.sin(a);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fillStyle   = "rgba(124,106,247,0.20)";
  ctx.strokeStyle = "#7c6af7";
  ctx.lineWidth   = 2;
  ctx.fill();
  ctx.stroke();

  // dots
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const rv = R * vals[i];
    const x = cx + rv * Math.cos(a);
    const y = cy + rv * Math.sin(a);
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fillStyle = "#7c6af7";
    ctx.fill();
  }

  // labels
  ctx.font      = "bold 13px 'Space Mono', monospace";
  ctx.fillStyle = "#e8eaf0";
  ctx.textAlign = "center";
  const labelR = R + 28;
  for (let i = 0; i < n; i++) {
    const a = (Math.PI * 2 * i) / n - Math.PI / 2;
    const x = cx + labelR * Math.cos(a);
    const y = cy + labelR * Math.sin(a) + 4;
    ctx.fillText(labels[i], x, y);
  }
}


// ─────────────────────────────────────────────
// Load Game Evaluation
// ─────────────────────────────────────────────
function loadGameEval() {
  const loadingEl = document.getElementById("eval-loading");
  const cardEl    = document.getElementById("eval-card");
  loadingEl.textContent = "Loading…";
  loadingEl.classList.remove("hidden");
  cardEl.classList.add("hidden");

  fetch(`${API}/api/game-evaluation`)
    .then(r => {
      if (!r.ok) throw new Error("No evaluation yet");
      return r.json();
    })
    .then(d => {
      loadingEl.classList.add("hidden");
      cardEl.classList.remove("hidden");

      // rating & summary
      document.getElementById("eval-rating").textContent  = d.rating || "—";
      document.getElementById("eval-summary").textContent = d.summary || "";

      // badges
      const bs = document.getElementById("badge-satisfied");
      bs.textContent = d.player_satisfied ? "✓ Players Satisfied" : "✗ Players Not Satisfied";
      bs.className   = "badge " + (d.player_satisfied ? "yes" : "no");

      const bl = document.getElementById("badge-learning");
      bl.textContent = d.learning_achieved ? "✓ Learning Achieved" : "✗ Learning Not Achieved";
      bl.className   = "badge " + (d.learning_achieved ? "yes" : "no");

      // dimension cards
      const dims = [
        { key: "PD", score: d.avg_PD, label: d.PD_label, rec: d.PD_recommendation },
        { key: "TD", score: d.avg_TD, label: d.TD_label, rec: d.TD_recommendation },
        { key: "LD", score: d.avg_LD, label: d.LD_label, rec: d.LD_recommendation },
        { key: "BD", score: d.avg_BD, label: d.BD_label, rec: d.BD_recommendation },
      ];

      dims.forEach(dim => {
        const k = dim.key.toLowerCase();
        setLabel(`${k}-label`,  dim.label);
        setBar(`${k}-bar`, dim.score);
        document.getElementById(`${k}-score`).textContent = dim.score.toFixed(3);
        document.getElementById(`${k}-rec`).textContent   = dim.rec || "";
      });

      // radar
      drawRadar({
        PD: d.avg_PD,
        TD: d.avg_TD,
        LD: d.avg_LD,
        BD: d.avg_BD,
      });
    })
    .catch(err => {
      loadingEl.textContent = "⚠ " + err.message;
    });
}


// ─────────────────────────────────────────────
// Load Fuzzy Player Table
// ─────────────────────────────────────────────
function loadFuzzy() {
  const loadEl  = document.getElementById("fuzzy-loading");
  const wrapEl  = document.getElementById("fuzzy-table-wrap");
  const bodyEl  = document.getElementById("fuzzy-body");
  loadEl.textContent = "Loading…";
  loadEl.classList.remove("hidden");
  wrapEl.classList.add("hidden");

  fetch(`${API}/api/fuzzy-results`)
    .then(r => r.json())
    .then(data => {
      loadEl.classList.add("hidden");
      if (!data.length) { loadEl.textContent = "No data yet."; loadEl.classList.remove("hidden"); return; }

      wrapEl.classList.remove("hidden");
      bodyEl.innerHTML = data.map(p => `
        <tr>
          <td><strong>${p.player_id}</strong></td>
          <td>
            ${scorePill(p.PD)}
            <span style="font-size:10px;color:var(--text-muted);margin-left:4px">${p.PD_label || ""}</span>
          </td>
          <td>
            ${scorePill(p.TD)}
            <span style="font-size:10px;color:var(--text-muted);margin-left:4px">${p.TD_label || ""}</span>
          </td>
          <td>
            ${scorePill(p.LD)}
            <span style="font-size:10px;color:var(--text-muted);margin-left:4px">${p.LD_label || ""}</span>
          </td>
          <td>
            ${scorePill(p.BD)}
            <span style="font-size:10px;color:var(--text-muted);margin-left:4px">${p.BD_label || ""}</span>
          </td>
          <td>${scorePill(p.fuzzy_score)}</td>
          <td class="${levelClass(p.ML_level)}">${p.ML_level || "—"}</td>
        </tr>
      `).join("");
    })
    .catch(err => {
      loadEl.textContent = "Error: " + err.message;
      loadEl.classList.remove("hidden");
    });
}


// ─────────────────────────────────────────────
// Load ML Clustering Table
// ─────────────────────────────────────────────
function loadML() {
  const loadEl  = document.getElementById("ml-loading");
  const wrapEl  = document.getElementById("ml-table-wrap");
  const bodyEl  = document.getElementById("ml-body");
  loadEl.textContent = "Loading…";
  loadEl.classList.remove("hidden");
  wrapEl.classList.add("hidden");

  fetch(`${API}/api/ml-results`)
    .then(r => r.json())
    .then(data => {
      loadEl.classList.add("hidden");
      if (!data.length) { loadEl.textContent = "No data yet."; loadEl.classList.remove("hidden"); return; }

      wrapEl.classList.remove("hidden");
      bodyEl.innerHTML = data.map(p => `
        <tr>
          <td><strong>${p.player_id}</strong></td>
          <td class="${levelClass(p.level)}">${p.level || "—"}</td>
          <td>${p.cluster !== undefined ? p.cluster : "—"}</td>
          <td>${p.model || "—"}</td>
          <td>${scorePill(p.score)}</td>
        </tr>
      `).join("");
    })
    .catch(err => {
      loadEl.textContent = "Error: " + err.message;
      loadEl.classList.remove("hidden");
    });
}
