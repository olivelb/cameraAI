# CLI de contrôle pour un LLM

Le CLI contrôle **la même instance native** que l’utilisateur. Il ne lance pas
un deuxième lecteur vidéo. Les modifications sont visibles dans l’interface.
Utiliser le Python de l’environnement de l’application.

```powershell
python camera_cli.py schema.get
python camera_cli.py app.start
python camera_cli.py status.get
'{"enabled":true,"tracker":"ByteTrack"}' | python camera_cli.py tracking.set --stdin
'{"message":"Décris cette image en français."}' | python camera_cli.py assistant.ask --stdin --timeout 600
```

`schema.get` fonctionne même si l’application est fermée et décrit les 38 commandes
et leurs paramètres au format JSON Schema. C’est le point d’entrée conseillé
pour un agent : lire le catalogue, lire `status.get`, exécuter une opération,
puis vérifier son résultat et l’état courant.

## Contrat JSON et tâches asynchrones

Chaque commande écrit un seul objet JSON UTF-8 sur stdout. Les logs de lancement
vont dans `cli-launch.log`, ignoré par Git.

| Code de sortie | Sens |
| --- | --- |
| 0 | Opération réussie, ou tâche explicitement acceptée avec `--no-wait` |
| 1 | Échec de l’opération, ou délai d’attente du CLI dépassé |
| 2 | Paramètres invalides ou application inaccessible |

Les réponses portent `ok`, puis `result` ou `error`. Les opérations exécutées par
le worker ont un `job_id`. Une erreur du moteur, y compris une erreur VLM, donne
`ok:false` : elle n’est pas transformée en réponse réussie.

```powershell
$job = '{"model":"yolo26n.pt","task":"Segmentation"}' |
  python camera_cli.py model.load --stdin --no-wait | ConvertFrom-Json
@{id=$job.job.id} | ConvertTo-Json -Compress |
  python camera_cli.py job.get --stdin
python camera_cli.py jobs.list
python camera_cli.py status.get
```

Sans `--no-wait`, le CLI attend au plus 120 secondes ; `--timeout` permet de
choisir entre 0,1 et 3600 secondes. `WAIT_TIMEOUT` signifie que **la tâche continue**.
Interroger `job.get` avec le même identifiant ; ne pas renvoyer aveuglément une
commande de chargement, d’export ou d’écriture. Les résultats sont conservés en
mémoire pour les 128 dernières tâches, avec au plus 32 commandes en attente.
Le statut et les options d’affichage restent accessibles pendant une tâche longue.
Le premier chargement de l’assistant peut prendre plusieurs minutes ; utiliser
un délai adapté ou `--no-wait`, puis suivre la tâche avec `job.get`.

## Commandes par fonctionnalité

Les paramètres exacts, types, choix et bornes viennent de `schema.get`.

| Fonction | Commandes |
| --- | --- |
| Cycle de vie | `app.start`, `app.quit` |
| Catalogue et état | `schema.get`, `status.get`, `jobs.list`, `job.get` |
| Caméra | `camera.get`, `camera.set`, `camera.reconnect`, `camera.restart` |
| Réseaux et Tailscale | `network.get`, `network.use`, `network.configure`, `network.tailscale` |
| Détection / IA | `ai.set`, `detection.set` |
| Modèles et TensorRT | `model.get`, `model.load`, `model.export` |
| Concepts SAM3 | `sam.concepts` |
| Suivi | `tracking.set` |
| Reconnaissance | `recognition.set` avec `kind:faces` ou `kind:animals` |
| Galeries | `gallery.status`, `gallery.import`, `gallery.rebuild`, `gallery.learn`, `gallery.stop` |
| Alertes | `alerts.get`, `alerts.configure`, `alerts.enable`, `alerts.test` |
| Assistant | `assistant.ask`, `assistant.history`, `assistant.clear` |
| Capture PNG | `frame.save` |
| Aperçu / plein écran | `view.get`, `view.set` |
| Journal | `logs.get` |

`tracking.set` applique immédiatement l’activation et l’algorithme. Le statut
distingue `requested` et `effective`, et expose les derniers identifiants suivis.
Une liste d’identifiants vide peut simplement signifier qu’aucun objet n’est suivi.
SAM2/SAM3 utilisent la segmentation et ne fournissent pas ce suivi YOLO.
Les champs `frame_id` et `processed_id` permettent de vérifier qu’une nouvelle
image a effectivement été reçue et traitée après une commande.

`gallery.learn` démarre la collecte ; la fin de sa tâche signifie que la demande
est enregistrée. Les compteurs `learning` dans `status.get` décrivent la progression
réelle. Une capture ratée ne décrémente plus le compteur. `gallery.stop` conserve
les photos déjà capturées. `frame.save` refuse d’écraser un fichier existant sans
`overwrite:true`.

## Email et données privées

`alerts.configure` sauvegarde des identifiants chiffrés sans envoyer d’email.
`alerts.test` sans paramètres valide leur présence, **sans envoi**.
`alerts.test` avec `send:true` envoie réellement un email ; `alerts.enable` avec
`enabled:true` active les envois automatiques. Un agent doit réserver ces deux
actions à une demande explicite de l’utilisateur.

Les mots de passe et jetons doivent être transmis avec `--stdin` ou `--file`,
jamais avec `--json` dans la ligne de commande. Les identifiants restent locaux.
Pour un fichier de paramètres, utiliser un emplacement privé ignoré par Git.

`network.get` masque les adresses par défaut ; `private:true` les révèle
explicitement, sans jamais retourner de jeton. `network.tailscale` expose la liste
locale des appareils, et `assistant.history` le texte de la conversation : leurs
sorties sont privées. `logs.get` masque les détails de connexion et identifiants
par défaut. Ne pas publier les résultats réels de qualification.

Le canal HTTP écoute seulement sur la boucle locale, sur un port attribué au
démarrage. Il exige un secret aléatoire conservé dans `.camera-control.json`,
ignoré par Git et supprimé à la fermeture. Le CLI lit ce fichier automatiquement.
Il désactive les proxies HTTP pour ce canal. `--state-file` permet de choisir un
descripteur privé distinct ; `desktop.py --no-control` désactive le canal.

## Tests et qualification isolée

```powershell
python -m pytest tests/test_cli.py -q
python -m pytest -q
python -m ruff check .
```

Les tests exécutent **chaque commande** dans un vrai sous-processus CLI, via HTTP,
la boucle Qt et son worker. Les frontières caméra/ML/SMTP sont simulées en CI ;
les écritures de configuration chiffrée, captures PNG, tâches, validation et
synchronisation de l’interface sont exercées réellement. D’autres tests couvrent
le préfixe du parseur VLM et l’activation directe du suivi.

Pour les essais matériels, lancer une instance isolée avec un autre dossier de
galeries/email et un autre descripteur, après avoir fermé le lecteur caméra :

```powershell
'{"data_dir":".audit-tmp/cli-data"}' |
  python camera_cli.py app.start --state-file .audit-tmp/control.json --stdin
python camera_cli.py status.get --state-file .audit-tmp/control.json
python camera_cli.py app.quit --state-file .audit-tmp/control.json
```

Créer le dossier parent du descripteur avant le lancement. Les modèles partagent
le cache de l’installation ; les galeries et identifiants email de production
restent séparés. L’option `data_dir` s’applique au démarrage d’une nouvelle
instance ; un descripteur déjà utilisé est refusé pour éviter de tester sur le
mauvais jeu de données. Pour un essai sans caméra/chargement IA, ajouter
`no_camera:true` au JSON. La fermeture attend les écritures et libère le flux.

Le serveur du Pi n’accepte qu’un lecteur vidéo à la fois. La réussite des tests
de contrat ne qualifie pas automatiquement tous les modèles, le réseau, le GPU
ou la précision de reconnaissance : les résultats matériels doivent être
consignés séparément, sans adresses ni informations personnelles.
