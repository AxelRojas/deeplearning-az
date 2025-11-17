import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import pickle

def csvFinancialDataSetCorrection(path, dataSet):
	dataSet.columns = dataSet.columns.str.strip()
	col_name = "State Mean Encoded"
	#Handling categorized data using Mean Encoding
	dataSet[col_name] = dataSet["State"].str.replace(" ", "", regex=False)
	dataSet[col_name] = dataSet["State"].str.strip().str.lower()
	dataSet[col_name] = dataSet[['R&D Spend', 'Administration', 'Marketing Spend']].mean(axis=1, skipna=True) #I will get mean of each of row and replace in State column
	col = dataSet.pop("State Mean Encoded")
	dataSet.insert(3, "State Mean Encoded", col)

	for col in dataSet.select_dtypes(include="number").columns:
		dataSet[col] = dataSet[col].clip(lower=0)
		media = dataSet[col].replace(0, np.nan).mean()
		dataSet[col] = dataSet[col].replace(0, np.nan).fillna(media)

		#yield dataSet[col]
	#print(dataSet) #Visualize csv corrected

def normalizationOfData(dataSet, scaler):
	num_cols = dataSet.select_dtypes(include="number").columns
	dataSet[num_cols] = scaler.fit_transform(dataSet[num_cols])

def saveWorkedDataSetAndScaler(path, dataSet, scaler):
	with open(path+"scaler.pkl", "wb") as f:
		pickle.dump(scaler, f)

	dataSet.to_csv(path+"50_Startups_CleandUp.csv", index=False)  # index=False para no guardar la columna de índices



if __name__ == "__main__":
	csvPath = "/Users/axeljavierrojasmosqueda/Documents/DeepLearning/deeplearning-az/personalProject/DataSets/50_Startups.csv"
	new_csv_path = "/Users/axeljavierrojasmosqueda/Documents/DeepLearning/deeplearning-az/personalProject/DataSets/"
	path = Path(csvPath)
	new_path = Path(new_csv_path)
	if path.exists():
		dataSet = pd.read_csv(path)
		csvFinancialDataSetCorrection(path, dataSet)
		scaler = MinMaxScaler()
		normalizationOfData(dataSet, scaler)
		#print(dataSet) #Visualize csv corrected
		saveWorkedDataSetAndScaler(new_csv_path, dataSet, scaler)
	else:
		raise FileNotFoundError
