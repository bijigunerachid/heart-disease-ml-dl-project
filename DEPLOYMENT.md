# 🚀 Guide de Déploiement - CardioRisk AI

## Déployer sur Streamlit Cloud (GRATUIT & FACILE)

### Prérequis
- Compte GitHub avec votre repo du projet
- Compte Streamlit Cloud (gratuit)

### Étapes

#### 1️⃣ Vérifier votre repo GitHub

```bash
git status
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

#### 2️⃣ Accéder à Streamlit Cloud

1. Allez sur : **https://share.streamlit.io**
2. Connectez-vous avec GitHub

#### 3️⃣ Déployer l'app

- Cliquez **"New app"**
- Remplissez :
  - Repository: `<votre-username>/heart-disease-ml-dl-project`
  - Branch: `main`
  - Main file: `app.py`
- Cliquez **"Deploy"**

✅ Votre app sera disponible à : `https://cardiorisk-ai.streamlit.app`

---

## 📊 Alternatives de déploiement

| Plateforme | Coût | Facilité | Modèles volumineux |
|---|---|---|---|
| **Streamlit Cloud** ⭐ | Gratuit | ⭐⭐⭐ | ✅ (jusqu'à 1GB) |
| Heroku | Payant | ⭐⭐ | ⚠️ |
| AWS EC2 | Payant | ⭐ | ✅ |
| Railway | Gratuit/Payant | ⭐⭐ | ⚠️ |

### 🎯 **Recommandation** : Utilisez **Streamlit Cloud** (c'est l'official, gratuit et optimisé pour Streamlit)

---

## 🔧 Commandes utiles

### Tester localement avant de déployer
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Voir les logs en production
Dans Streamlit Cloud Dashboard → Votre app → "Manage app" → Logs

---

## 📝 Notes importantes

1. ⚠️ **Modèles volumineux** : Stockez les fichiers `.pt` et `.joblib` directement dans le repo
2. 🔐 **Secrets** : Pas d'API keys ou credentials dans le code. Utilisez `.streamlit/secrets.toml`
3. 📦 **Dépendances** : Gardez `requirements.txt` à jour
4. ⚡ **Performance** : Streamlit Cloud a des limitations (CPU/RAM). Pour plus de puissance → AWS

---

## ✨ Votre app sera en LIVE en moins de 5 minutes !
