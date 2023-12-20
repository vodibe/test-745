# MLFlow

Con MLFlow noi possiamo monitorare e interagire il ciclo di vita di un progetto ML.

## Ciclo di vita

![ciclo_di_vita_ml](../docs/mlflow-overview.png)

**Data preparation**: ricerchiamo, raccogliamo, monitoriamo dati, risorse, esperimenti
correlate al nostro progetto ML. Risorse sono "grezze".

**Data analysis esplorativa**: indichiamo in che contesto si pone il nostro progetto ML:
che problema risolve, quale task rende più efficiente. Magari spearando le aree geografiche in cui i dati si riferiscono, analizzando  delle prime "tendenze" tra i dati raccolti...

**Feature engineering**: inizia la fase di "purificazione" dei dati raccolti. Ingegnerizziamo
nuove feature, rimuoviamo alcune. Studiamo il dominio di ciascuna feature ecc...

**Model training**: quali task di classificazione/regressione si intende fare. Quali sono i
modelli che meglio possono gestire i dati. Fase che TIENE CONTO SOLAMENTE DEGLI ESEMPI "OFFLINE".
Magari si presume che quando il modello verrà pubblicato e reso pronto per gli esempi "ONLINE",
questi siano distribuiti come gli esempi "OFFLINE". Ma non abbiamo la certezza.

**Model validation**: calcolo delle performance dei modelli addestrati. Confronto.

**Deployment**: il modello viene messo in esecuzione nell'ambiente che è quello utilizzato dall'organizzazione. Non ci aspetteremo che l'ambiente su cui il modello girerà sarà lo stesso ambiente nel quale è stato sviluppato. Possono cambiare le versioni delle librerie (perchè magari un software dell' organizzazione è stato sviluppato su questa versione specifica). Si parla di **stage**, cioè un' operazione automatizzata di recupero librerie, configurazioni ecc.. (approfondire). Si può creare un'astrazione del modello, cioè può essere visto come una funzione
python. 

**Monitoring**: Come si comporta il modello sulle istanze "online"? Operazione fondamentale perchè
ad esempio quando si verificano avvenimenti importanti (pandemie ecc...) gli esempi online possono
cambiare distribuzione. Un modello che era risultato ottimo, ora può diventare inutile.

## MLflow Tracking (server + client grafico)

https://mlflow.org/docs/latest/getting-started/intro-quickstart/index.html

```
cd mlflow_lab
```
```
mlflow server --host 127.0.0.1 --port 8080
```
I concetti che ci interessano sono:

1 ) - Cosa vogliamo fare con un modello di apprendimento? Ci serve un modello di classificazione o regressione? **CONCETTO DI ESPERIMENTO**
```
mlflow.set_experiment("MLflow Quickstart")
# se non decidiamo un esperimento, ci sarà un esperimento di default.
```

2 ) - Creiamo il modello, es. un oggetto di una classe offerta da scikitlearn. Es. `/models/regr_logistic.py`. **ASSOCIATO AL MODELLO CREATO C'E' IL CONCETTO DI RUN**: esecuzioni di alcune parti di codice che genera dei metadati:
- combinazione di iperparametri
- metriche
- orari di inizio e fine
- artefatti:
  - file di output della run (pesi del modello, immagini, ecc.).
  
NOTA: Se più run utilizzano lo stesso dataset di input (anche se ne utilizzano parti diverse), appartengono logicamente allo stesso esperimento. Per altre categorizzazioni gerarchiche è consigliabile l'utilizzo dei tag.
```
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    signature = infer_signature(X_train, lr.predict(X_train))
    ...
```
![](../docs/Immagine.png)

![](../docs/training-annotation.svg)

3 ) - A ogni run c'è una firma = comportamento del modello = TS + predizioni

4 ) - **MLflow Model** = modello + firma

- Se il modello è un oggetto di una classe offerta da scikitlearn:
```
    model1_info = mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="iris_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="tracking-quickstart",
    )
```

5 ) - Per astrarre il modello creiamo  `/models_pyfunc/regr_logistic_pyfunc.py`.

```
loaded_model1 = mlflow.pyfunc.load_model(model1_info.model_uri)

predictions = loaded_model1.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_feature_names)

result["actual_class"] = y_test
result["predicted_class"] = predictions
```

6 )  - Eseguiamo il codice, e ci accorgeremo che verrà creata 1 run (e annesso esperimento). Verifichiamo nell'interfaccia web `localhost:8080`
![mlflow_esperimento](../docs/quickstart-our-run.png)

## MLflow Tracking (server + client programmato)

https://mlflow.org/docs/latest/getting-started/logging-first-model/step2-mlflow-client.html

6 )  - Eseguiamo il codice, e ci accorgeremo che verrà creata 1 run (e annesso esperimento). Verifichiamo programmaticamente.
```
from mlflow import MlflowClient
from pprint import pprint
from sklearn.ensemble import RandomForestRegressor

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
all_experiments = client.search_experiments()

print(all_experiments)
> [
    <Experiment:
        artifact_location='./mlruns/0',
        creation_time=None,
        experiment_id='0',
        last_update_time=None,
        lifecycle_stage='active',
        name='Default',
        tags={}
    >
]
```
![tag_experiment_run](../docs/tag-exp-run-relationship.svg)

```
# Provide an Experiment description that will appear in the UI
experiment_description = (
    "This is the grocery forecasting project. "
    "This experiment contains the produce models for apples."
)

# Provide searchable tags that define characteristics of the Runs that
# will be in this Experiment
experiment_tags = {
    "project_name": "grocery-forecasting",
    "store_dept": "produce",
    "team": "stores-ml",
    "project_quarter": "Q3-2023",
    "mlflow.note.content": experiment_description,
}

# Create the Experiment, providing a unique name
produce_apples_experiment = client.create_experiment(
    name="Apple_Models", tags=experiment_tags
)
```
![experiment_ui](../docs/experiment-page-elements.svg)

Cosa possiamo fare programmaticamente:
- cercare esperimenti a partire da tag
- https://mlflow.org/docs/latest/tracking/tracking-api.html

## MLflow Tracking (server + database)

https://mlflow.org/docs/latest/tracking/backend-stores.html














  






