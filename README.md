
# Análisis de Sentimientos de Twitter

Este proyecto se enfoca en llevar a cabo un análisis de sentimientos en tweets con el propósito de clasificarlos como positivos, negativos o neutrales.

## Requisitos del Sistema

Para ejecutar este proyecto, asegúrate de tener instalado Python 3.x. Puedes descargarlo [aquí](https://www.python.org/downloads/).

## Librerías Utilizadas

- pandas
- seaborn
- numpy
- re
- string
- nltk
- matplotlib
- scikit-learn

Puedes instalar las librerías necesarias utilizando el siguiente comando:

```bash
pip install pandas seaborn numpy re nltk matplotlib scikit-learn
```

## Instrucciones de Uso

1. Clona o descarga este repositorio.
2. Asegúrate de tener el archivo `Twitter Sentiments.csv` en la misma carpeta que este archivo.
3. Ejecuta el código proporcionado en tu entorno de Python.

## Estructura del Proyecto

- `Twitter Sentiments.csv`: Contiene los datos de los tweets.
- `Análisis de Sentimientos.ipynb`: Notebook de Jupyter con el código y el análisis.
- `README.md`: Documentación del proyecto.
- Otros archivos y directorios según sea necesario.

## Código Destacado

```python
# ... (incluye tu código más relevante)

# Cargar datos
df = pd.read_csv("Twitter Sentiments.csv")

# Limpiar y procesar tweets
df["twiter limpiado"] = np.vectorize(remplazar)(df["tweet"],"@[\w]*")
df["twiter limpiado"] = df["twiter limpiado"].apply(lambda x: re.sub(r'[^a-zA-Z#\s]', '', x))
# ...

# Análisis de sentimientos con Regresión Logística
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['twiter limpiado'])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'])

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

lr = LogisticRegression()
lr.fit(x_train, y_train)

ypred = lr.predict(x_test)

accuracy_score(ypred, y_test)
```

## Contribuciones

Si deseas contribuir a este proyecto, por favor abre un problema o envía una solicitud de extracción.

## Licencia

Este proyecto está licenciado bajo la [Licencia MIT](LICENSE).

