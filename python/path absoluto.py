
import pandas as pd
df = pd.read_csv('../archivos/csv/paginaCorrespondiente.csv', sep=',')
clase='videoconferencia'
tema=str(5)
print('../archivos/csv/paginaCorrespondiente.csv')
print("Clase: "+clase+"\nTema: "+str(tema))
print(df.head())
subclase = df.loc[:, 'SubClass']==int(tema)
print(subclase)
df_2 = df.loc[subclase]
print(df_2.head())
pagina = df_2.values[0][2]
print("PAGINA  ", pagina)

"""
import pandas as pd
df = pd.read_csv('../archivos/csv/paginaCorrespondiente.csv', sep=',')
clase='videoconferencia'
tema=str(5)
print('../archivos/csv/paginaCorrespondiente.csv')
print("Clase: "+clase+"\nTema: "+str(tema))
print(df.head())
clase = df.loc[:, 'Class'] == clase
df_1 = df.loc[clase]
print(df_1.head())
subclase = df_1.loc[:, 'SubClass'] == tema
df_2 = df_1.loc[subclase]
print(df_2.head())
pagina = df_2.values[0][2]
print("PAGINA  ", pagina)"""