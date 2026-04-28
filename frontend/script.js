const API = "http://localhost:5000";

// ── Helpers ──────────────────────────────────────────────────────────────────

function showError(containerId, message) {
  const el = document.getElementById(containerId);
  if (el) el.innerHTML = `<p class="error">❌ ${message}</p>`;
}

function setTableBody(tableId, html) {
  const tbody = document.querySelector(`#${tableId} tbody`);
  if (tbody) tbody.innerHTML = html;
}

// ── Metrics ──────────────────────────────────────────────────────────────────

function loadMetrics() {
  fetch(`${API}/api/metrics`)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(d => {
      if (!d) {
        document.getElementById("metrics-container").innerHTML =
          "<p>No metrics available yet.</p>";
        return;
      }

      const date = d.timestamp
        ? new Date(d.timestamp * 1000).toLocaleString()
        : "—";

      document.getElementById("metrics-container").innerHTML = `
        <table>
          <thead><tr><th>Model</th><th>Silhouette Score</th></tr></thead>
          <tbody>
            <tr class="${d.best_model === 'kmeans' ? 'best' : ''}">
              <td>KMeans ${d.best_model === 'kmeans' ? '🏆' : ''}</td>
              <td>${(d.kmeans ?? "—").toFixed ? d.kmeans.toFixed(4) : d.kmeans}</td>
            </tr>
            <tr class="${d.best_model === 'dbscan' ? 'best' : ''}">
              <td>DBSCAN ${d.best_model === 'dbscan' ? '🏆' : ''}</td>
              <td>${(d.dbscan ?? "—").toFixed ? d.dbscan.toFixed(4) : d.dbscan}</td>
            </tr>
            <tr class="${d.best_model === 'agglo' ? 'best' : ''}">
              <td>Agglomerative ${d.best_model === 'agglo' ? '🏆' : ''}</td>
              <td>${(d.agglo ?? "—").toFixed ? d.agglo.toFixed(4) : d.agglo}</td>
            </tr>
          </tbody>
        </table>
        <p class="timestamp">Last updated: ${date}</p>
      `;
    })
    .catch(err => {
      console.error("Metrics ERROR:", err);
      showError("metrics-container", `Could not load metrics — ${err.message}`);
    });
}

// ── ML Results ────────────────────────────────────────────────────────────────

function loadML() {
  fetch(`${API}/api/ml-results`)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(data => {
      if (!data.length) {
        setTableBody("ml", '<tr><td colspan="5">No data yet.</td></tr>');
        return;
      }

      const rows = data.map(p => `
        <tr>
          <td>${p.player_id ?? "—"}</td>
          <td>${p.score ?? "—"}</td>
          <td><span class="badge badge-${p.level}">${p.level ?? "—"}</span></td>
          <td>${p.cluster ?? "—"}</td>
          <td>${p.model ?? "—"}</td>
        </tr>
      `).join("");

      setTableBody("ml", rows);
    })
    .catch(err => {
      console.error("ML ERROR:", err);
      setTableBody("ml", `<tr><td colspan="5" class="error">❌ ${err.message}</td></tr>`);
    });
}

// ── Fuzzy Results ─────────────────────────────────────────────────────────────

function loadFuzzy() {
  fetch(`${API}/api/fuzzy-results`)
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(data => {
      if (!data.length) {
        setTableBody("fuzzy", '<tr><td colspan="3">No data yet.</td></tr>');
        return;
      }

      const rows = data.map(p => `
        <tr>
          <td>${p.player_id ?? "—"}</td>
          <td>${p.fuzzy_score != null ? p.fuzzy_score.toFixed(4) : "—"}</td>
          <td><span class="badge badge-${p.decision}">${p.decision ?? "—"}</span></td>
        </tr>
      `).join("");

      setTableBody("fuzzy", rows);
    })
    .catch(err => {
      console.error("Fuzzy ERROR:", err);
      setTableBody("fuzzy", `<tr><td colspan="3" class="error">❌ ${err.message}</td></tr>`);
    });
}
