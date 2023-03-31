import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt

df = pd.read_excel("../estagio_dados/dados.xlsx")
df.head()

X = df[['nota_avaliacao','nota_ingles','nota_dinamica', 'nota_entrevista','monitoria', 'iniciacao_cientifica','vulnerabilidades_reportadas']]
y = df['Aprovado']

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, train_size= 0.70)

logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
y_pred = logistic_regression.predict(X_test)

confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
sn.heatmap(confusion_matrix, annot=True)

print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
plt.show()

teste =   {'nota_avaliacao': 8, 'nota_ingles': 9, 'nota_dinamica': 8, 'nota_entrevista': 7, 'monitoria': 0, 'iniciacao_cientifica':0, 'vulnerabilidades_reportadas':0}
dft = pd.DataFrame(data = teste, index=[0])
print(dft)

resultado = logistic_regression.predict(dft)
print(resultado)