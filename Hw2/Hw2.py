import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Hw2/Automobile_data_tiny.csv')

Mean = df.mean()
SD = df.std()

zscore = (df - Mean) / SD

print(zscore)

#part 2



calcCorrMatrix = df.corr()
print(calcCorrMatrix)


#part 3
eigenvalies,eigenvectors = np.linalg.eig(calcCorrMatrix)


print(eigenvectors[:,0])
print(eigenvectors[:,1])

#Question 3

#a Binning 2 groups together 

df = pd.read_csv("Hw2/WaterPol.csv")

#print(df.columns.tolist())


sortTheData = df.sort_values('Water Pollution',ascending=True)
#print(sortTheData)

mid = len(df) // 2
bin1 = sortTheData.iloc[:mid]
bin2 = sortTheData.iloc[mid:]

print(bin1)
print(bin2)


#Part b

minVal = df['Substance A'].min()
maxVal = df['Substance A'].max()
df['Substance A'] = (df['Substance A']-minVal) / (maxVal - minVal)



minValB = df['Substance B'].min()
maxValB = df['Substance B'].max()
df['Substance B'] = (df['Substance B']-minValB) / (maxValB - minValB)

sortTheData = df.sort_values('Water Pollution',ascending=True)
#print(sortTheData)

mid = len(df) // 2
bin1 = sortTheData.iloc[:mid]
bin2 = sortTheData.iloc[mid:]

print(bin1)
print(bin2)



# Getadata = bin1['Substance A']
# GetAMax = df['Substance A'].min()
# GetaMin = df['Substance A'].max()

# df['Normalised A'] = (df['Substance A'] - GetaMin) / (GetAMax- GetaMin)


#part c

#Group 1 A
plt.figure(figsize=(10,10))
plt.scatter(bin1.index,bin1['Substance A'])
plt.xlabel('Experiment')
plt.ylabel('Substance A Normilzed ')
plt.title('Bin 1- Substance A')
plt.show()

#Group 1 B
plt.figure(figsize=(10,10))
plt.scatter(bin1.index,bin1['Substance B'])
plt.xlabel('Experiment')
plt.ylabel('Substance B Normilzed ')
plt.title('Bin 1- Substance B')
plt.show()

#Group 2 A
plt.figure(figsize=(10,10))
plt.scatter(bin2.index,bin2['Substance A'])
plt.xlabel('Experiment')
plt.ylabel('Substance B Normilzed ')
plt.title('Bin 2- Substance B')
plt.show()

#Group 2 B
plt.figure(figsize=(10,10))
plt.scatter(bin2.index,bin2['Substance B'])
plt.xlabel('Experiment')
plt.ylabel('Substance B Normilzed ')
plt.title('Bin 2- Substance B')
plt.show()

CorrelationCofbin1 = bin1[['Substance A','Substance B']].corr().iloc[0,1]

CorrelationCofbin2 = bin2[['Substance A','Substance B']].corr().iloc[0,1]

print(CorrelationCofbin1)
print(CorrelationCofbin2)