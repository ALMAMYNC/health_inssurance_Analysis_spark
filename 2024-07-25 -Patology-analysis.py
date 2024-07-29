# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC ## Etude des patients d'une assurance

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Chargement du dataset

# COMMAND ----------

# DBTITLE 1,Importation des packages
from pyspark.sql.functions import sum, col
from pyspark.ml.feature import Imputer

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

# DBTITLE 1,Chargement du dataset
# Le chemin d'accès au dataset
file_location = "/FileStore/tables/effectifs-1.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ";"

# Le chargement du dataset
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)




# COMMAND ----------

# DBTITLE 1,Aperçu des données
df.display(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Exploration des données

# COMMAND ----------

# MAGIC %md
# MAGIC * L'effectif total

# COMMAND ----------

# DBTITLE 1,Le nombre de ligne du dataset
df.count()

# COMMAND ----------

# DBTITLE 1,Les variables et leurs type
df.printSchema()

# COMMAND ----------

# DBTITLE 1,Les caractéristiques/variables
df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Netoyage du dataset

# COMMAND ----------

# DBTITLE 1,Suppression de la variable tri
# Suppression de la variable tri qui est moyen important
df = df.drop("tri")
df.columns

# COMMAND ----------

# MAGIC %md
# MAGIC #### - Gestion des valeurs manquantes

# COMMAND ----------

# DBTITLE 1,Vérification des valeurs manquantes
# Verifier les valeurs manquantes
missing_values_count = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])
missing_values_count.show()

# COMMAND ----------

# MAGIC %md
# MAGIC - Il y a 483 840 valeurs manquantes dans colonne patho_niv2, 1 048 320 dans la colonne, 1 238 024 dans ntop et prev et 60 480

# COMMAND ----------

# DBTITLE 1,Traitement des valeurs manquantes pour les variables numériques
# Imputation par la moyenne de la colonne

# Spécifier les colonnes à imputer
imputer = Imputer(inputCols=['Ntop', 'prev'], outputCols=["Ntop_imputed", "prev_imputed"])

# Appliquer l'imputation
imputer_model = imputer.fit(df)
data_imputed = imputer_model.transform(df)

# Afficher un échantillon des données imputées
data_imputed.show(5)

# COMMAND ----------

# DBTITLE 1,Suppression des anciennes colonnes
# Supprimer les anciennes colonnes
data_imputed = data_imputed.drop('Ntop', 'prev')

# Renommer les colonnes imputées
data_imputed = data_imputed.withColumnRenamed("Ntop_imputed", "Ntop") \
                           .withColumnRenamed("prev_imputed", "prev")

# Afficher un échantillon des données finales
data_imputed.show(5)

# COMMAND ----------

# DBTITLE 1,Imputation des valeurs manquantes pour les colonnes catégorielles
# Imputation des valeurs manquantes pour les colonnes catégorielles

# Trouver la valeur la plus fréquente pour la colonne catégorielle 'patho_niv2'
mode_patho_niv2_row = df.groupBy("patho_niv2").count().orderBy("count", ascending=False).first()

# Si mode_patho_niv2_row est None, choisir une valeur par défaut pour l'imputation
if mode_patho_niv2_row is not None:
    mode_patho_niv2 = mode_patho_niv2_row[0]
else:
    mode_patho_niv2 = "Unknown"  # ou une autre valeur par défaut appropriée



# Vérifier que la colonne 'patho_niv2' est présente dans le DataFrame et appliquer l'imputation
if 'patho_niv2' in data_imputed.columns and mode_patho_niv2 is not None:
    data_imputed = data_imputed.fillna({"patho_niv2": mode_patho_niv2})
else:
    print("La colonne 'patho_niv2' n'est pas présente dans le DataFrame ou mode_dept est None.")


# Afficher un échantillon des données finales
data_imputed.show(5)


# COMMAND ----------

# DBTITLE 1,Imputation des valeurs manquantes pour les colonnes catégorielles
# Trouver la valeur la plus fréquente pour la colonne catégorielle 'patho_niv3'
mode_patho_niv3_row = df.groupBy("patho_niv3").count().orderBy("count", ascending=False).first()

# Si mode_patho_niv3_row est None, choisir une valeur par défaut pour l'imputation
if mode_patho_niv3_row is not None:
    mode_patho_niv3 = mode_patho_niv2_row[0]
else:
    mode_patho_niv3 = "Unknown"  # ou une autre valeur par défaut appropriée


# Vérifier que la colonne 'patho_niv3' est présente dans le DataFrame et appliquer l'imputation
if 'patho_niv3' in data_imputed.columns and mode_patho_niv3 is not None:
    data_imputed = data_imputed.fillna({"patho_niv3": mode_patho_niv3})
else:
    print("La colonne 'patho_niv3' n'est pas présente dans le DataFrame ou mode_dept est None.")

data_imputed.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### - Gestion des doublons

# COMMAND ----------

# DBTITLE 1,Vérification des doublons
data_imputed.groupBy(data_imputed.columns).count().filter("count > 1").show()

# COMMAND ----------

# MAGIC %md
# MAGIC Pas de doublons dans le dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Statistiques descriptive

# COMMAND ----------

data_imputed.columns

# COMMAND ----------

# DBTITLE 1,Statistiques descriptives des données numériques
# Statistiques descriptives
data_imputed['Ntop','Npop','prev'].describe().show()

# COMMAND ----------

# DBTITLE 1,Distribution des pathologies
# Distribution des pathologies
data_imputed.groupBy('patho_niv1').count().orderBy('count', ascending=False).show()



# COMMAND ----------

# DBTITLE 1,Distribution des traitements
# Distribution des traitements
data_imputed.groupBy('top').count().orderBy('count', ascending=False).show()



# COMMAND ----------

# DBTITLE 1,Distribution des épisodes de soins par sexe
# Distribution des épisodes de soins par sexe
data_imputed.groupBy('sexe', 'patho_niv1').count().orderBy('count', ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### - Analyse des effectif
# MAGIC

# COMMAND ----------

# DBTITLE 1,Analyser les effectifs des patient  par pathologie, sexe, région, etc.
# Effectif  moyens des patients par pathologie
data_imputed.groupBy('patho_niv1').agg({'Ntop': 'mean'}).orderBy('avg(Ntop)', ascending=False).show()
# Effectif moyens des patients par région
data_imputed.groupBy('region').agg({'Ntop': 'mean'}).orderBy('avg(Ntop)', ascending=False).show()
# Effectifs moyens des patients par sexe
data_imputed.groupBy('sexe').agg({'Ntop': 'mean'}).orderBy('avg(Ntop)', ascending=False).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### -  Analyse de la Prévalence

# COMMAND ----------

# DBTITLE 1,Étudier la prévalence des pathologies par région, sexe, et classe d’âge pour identifier les tendances épidémiologiques.
# Prévalence par région
data_imputed.groupBy('region').agg({'prev': 'mean'}).orderBy('avg(prev)', ascending=False).show()

# Prévalence par sexe
data_imputed.groupBy('sexe').agg({'prev': 'mean'}).orderBy('avg(prev)', ascending=False).show()

# Prévalence par classe d’âge
data_imputed.groupBy('cla_age_5').agg({'prev': 'mean'}).orderBy('avg(prev)', ascending=False).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segmentation des Patients
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,Utiliser des techniques de clustering pour segmenter les patients en groupes homogènes basés sur leurs caractéristiques.
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Préparer les données
assembler = VectorAssembler(inputCols=['annee', 'sexe', 'region', 'Ntop', 'Npop', 'prev'], outputCol="features")
data_assembled = assembler.transform(data_imputed)

# K-Means Clustering
kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(data_assembled.select('features'))

# Prédictions
predictions = model.transform(data_assembled)

# Montrer les clusters
predictions.groupBy('prediction').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preparation des données pour la construction des modèles de prédiction 

# COMMAND ----------

# DBTITLE 1,Préparer les données : indexer les colonnes catégorielles et assembler les caractéristiques
# Préparer les données : indexer les colonnes catégorielles et assembler les caractéristiques
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in data_imputed.columns]
assembler = VectorAssembler(inputCols=['annee', 'patho_niv1_index', 'patho_niv2_index', 'patho_niv3_index', 'top_index', 'cla_age_5_index', 'sexe', 'region', 'dept_index', 'Ntop', 'Npop', 'prev', 'niveau_prioritaire_index', 'libelle_classe_age_index', 'libelle_sexe_index'], outputCol="features")


# COMMAND ----------

# MAGIC %md
# MAGIC ### - Segmentation des Patients
# MAGIC

# COMMAND ----------

# DBTITLE 1,Utiliser des techniques de clustering pour segmenter les patients en groupes homogènes basés sur leurs caractéristiques.

#Utiliser des techniques de clustering pour segmenter les patients en groupes homogènes basés sur leurs caractéristiques.

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans

# Préparer les données
assembler = VectorAssembler(inputCols=['annee', 'sexe', 'region', 'Ntop', 'Npop', 'prev'], outputCol="features")
data_assembled = assembler.transform(data_imputed)

# K-Means Clustering
kmeans = KMeans().setK(5).setSeed(1)
model = kmeans.fit(data_assembled.select('features'))

# Prédictions
predictions = model.transform(data_assembled)

# Montrer les clusters
predictions.groupBy('prediction').count().show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### - Separation des données en ensemble d'entrainement et de test

# COMMAND ----------

# Diviser les données en train et test
(trainingData, testData) = data_imputed.randomSplit([0.8, 0.2], seed=1234)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Construire un model de régression pour la classification des patient, a fin de pouvoir predire sa pathologie 

# COMMAND ----------

# DBTITLE 1,Créer le modèle RandomForest pour la classification des patients
# Créer le modèle RandomForest
rf = RandomForestClassifier(labelCol="sexe", featuresCol="features", numTrees=10, maxBins=110)

# Construire le pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Entraîner le modèle
model = pipeline.fit(trainingData)

# Faire des prédictions
predictions = model.transform(testData)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluation du model de randmForest

# COMMAND ----------

# Évaluer le modèle
evaluator = MulticlassClassificationEvaluator(labelCol="sexe", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Construire un model de régression pour prédire la prévalence

# COMMAND ----------

# DBTITLE 1,Création d'un model de regression
# Créer le modèle RandomForestRegressor avec maxBins augmenté
rf = RandomForestRegressor(labelCol="prev", featuresCol="features", numTrees=10, maxBins=110)

# Construire le pipeline
pipeline = Pipeline(stages=indexers + [assembler, rf])

# Entraîner le modèle
model = pipeline.fit(trainingData)

# Faire des prédictions
predictions = model.transform(testData)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluation du model de Régression pour la prédiction de la prévalence

# COMMAND ----------

# DBTITLE 1,Évaluer le modèle du model de regression
# Évaluer le modèle du model de regression
evaluator = RegressionEvaluator(labelCol="prev", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")

