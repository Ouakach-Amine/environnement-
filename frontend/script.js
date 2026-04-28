const API = "http://localhost:5000";

function loadML() {
  console.log("CLICK ML");

  fetch(`${API}/api/ml-results`)
    .then(r => r.json())
    .then(data => {
      console.log("ML DATA:", data);

      let html = "<tr><th>ID</th><th>Level</th></tr>";
      data.forEach(p => {
        html += `<tr><td>${p.player_id}</td><td>${p.level}</td></tr>`;
      });

      document.getElementById("ml").innerHTML = html;
    })
    .catch(err => console.error("ML ERROR:", err));
}

function loadFuzzy() {
  console.log("CLICK FUZZY");

  fetch(`${API}/api/fuzzy-results`)
    .then(r => r.json())
    .then(data => {
      console.log("Fuzzy DATA:", data);

      let html = "<tr><th>ID</th><th>Score</th></tr>";
      data.forEach(p => {
        html += `<tr><td>${p.player_id}</td><td>${p.fuzzy_score}</td></tr>`;
      });

      document.getElementById("fuzzy").innerHTML = html;
    })
    .catch(err => console.error("Fuzzy ERROR:", err));
}
