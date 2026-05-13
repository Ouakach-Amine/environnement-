# Guide de Configuration Multi-Environnement

## 🐳 Services Docker (Zookeeper, Kafka, MongoDB, Spark)

Pour démarrer les services Docker :

```bash
docker-compose up
```

**Ports exposés :**
- Zookeeper: `2181`
- Kafka: `9093`
- MongoDB: `27018` (mapped to 27017 in container)

---

## 🖥️ Services à Lancer en Terminaux Normaux

### 1️⃣ ML-Service (Terminal 1)

```bash
cd ml-service
pip install -r requirements.txt
python app.py
```

**Requis avant :** MongoDB et Kafka doivent être running (via Docker)

---

### 2️⃣ Backend (Terminal 2)

```bash
cd backend
pip install -r requirements.txt
python app.py
```

**Port par défaut :** `5000`
**Requis avant :** MongoDB et Kafka doivent être running (via Docker)

---

### 3️⃣ Frontend (Terminal 3)

```bash
cd frontend
npm install
npm start
```

**Port par défaut :** `3000`

---

## 📋 Ordre de Démarrage Recommandé

1. **Démarrer les conteneurs Docker** :
   ```bash
   docker-compose up
   ```
   ⏳ Attendre que MongoDB soit ready (healthcheck)

2. **Terminal 1 - ML-Service** :
   ```bash
   cd ml-service && pip install -r requirements.txt && python app.py
   ```

3. **Terminal 2 - Backend** :
   ```bash
   cd backend && pip install -r requirements.txt && python app.py
   ```

4. **Terminal 3 - Frontend** :
   ```bash
   cd frontend && npm install && npm start
   ```

---

## 🔗 Configuration des Connexions

Mettez à jour les fichiers de configuration pour pointer vers les services Docker :

- **MongoDB :** `mongodb://localhost:27018` (au lieu de la connexion Docker interne)
- **Kafka :** `localhost:9093` (au lieu de kafka:9092 depuis Docker)
- **Zookeeper :** `localhost:2181`

