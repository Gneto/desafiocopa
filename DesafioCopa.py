#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:34:49 2018

@author: orlando
"""

import pandas as pd
import numpy as np
import collections
import csv
from sklearn import linear_model
from sklearn import metrics

#carrega os datasets
train=pd.read_csv('dataset.csv', sep=';') # contem a base de historico de todos o jogos entre selecoes na copa
test=pd.read_csv('datasetTest.csv', sep=';') #contem a tabela da fase de grupo copa do mundo Russia 2018
teams=pd.read_csv('teams.csv', sep=';') #contem a base de selecoes, seu ranking se esta classificada para copa da Russia ou nao
copa=pd.read_csv('./Datasets/Copa.csv', sep=';') #historico de campeoes de copas do mundo e sua sede
grupoTime=pd.read_csv('grupoTime.csv', sep=';') # contem informacoes dos grupos da copa do mundo Russia 2018
jogadores=pd.read_csv('jogadores.csv', sep=';') #contem base dos jogadores convocados de cada selecao para a copa Russia 2018
cartoes=pd.read_csv('cartoes.csv', sep=';') #contem base historica de cartoes por selecao
cartoesTest=pd.read_csv('cartoesTest.csv', sep=';') #contem base de teste de cartoes para cada jogo

######################################################################
### Contem funcoes relativas ao time
### Eh muito util para o funcionamento do programa
######################################################################

# 10 melhores times pelo raking da fifa
list10BestTeams=['Germany', 'Brazil', 'Belgium', 'Portugal', 'Argentina', 'Switzerland', 
'France', 'Poland', 'Peru', 'Spain']

listCabecaChave=['Germany', 'Brazil', 'Belgium', 'Portugal', 'Argentina', 'Switzerland', 'France', 'Russia']


def getRankFifa(team_id, teams):
    return int(teams[teams['id'] == team_id].values[0][3])
    
def getTeamName(team_id, teams):
    return teams[teams['id'] == team_id].values[0][1]

def getTeamID(team_name, teams):
    return teams[teams['nome'] == team_name].values[0][0]

# funcao para checar se um time esta dentro do grupo das 10 melhores selecoes pelo ranking da fifa
def checkBestTeam(team_id, teams):
    teamName = getTeamName(team_id, teams)
    if (teamName in list10BestTeams):
        return 1
    else:
        return 0
        
def checkCabecaChave(team_id, teams):
    teamName = getTeamName(team_id, teams)
    if(teamName in listCabecaChave):
        return 1
    else:
        return 0
    
def checkChampion(team_id, year, copa, teams):
    year_champ = copa[copa['Ano'] == year]
    champ = year_champ['Campeao'].values
    if getTeamName(team_id, teams) == champ:
        return 1
    else:
        return 0
        
# Numero de campeonatos uma selecao tem
def getNumMundiaisTeam(team_id, teams, copa):
    campeoesList = copa['Campeao'].tolist()
    name = getTeamName(team_id, teams)
    return campeoesList.count(name)

######################################################################
### FIM - Das funcoes relativas ao time
######################################################################

###################################################################################
######## Pre processamento e preparacao dos dados para a analise
###################################################################################
#para os paises que n tiver ranking fifa setar 0
teams['rnk'].fillna(0, inplace=True)
#para os paises que n tiver participacao em copa setar 0
teams['participacoes_copa'].fillna(0, inplace=True)


#funcao responsavel por montar uma base historica de treinamento para test e train
# esta base servira para definar qual time irah fazer mais gols na copa
def createGols(dataset, typeDataset):
    if typeDataset == 'Train':
        golsTimeCasa=dataset.filter(items=['ano', 'id_casa', 'gols_casa'])
        golsTimeCasa=golsTimeCasa.rename(columns={'id_casa': 'time', 'gols_casa': 'gols'})
        golsTimeFora=dataset.filter(items=['ano', 'id_fora', 'gols_fora'])
        golsTimeFora=golsTimeFora.rename(columns={'id_fora': 'time', 'gols_fora': 'gols'})
        golsPartidas=pd.concat([golsTimeCasa, golsTimeFora])
        golsPartidas=golsPartidas.groupby(['time', 'ano'], as_index=False).gols.sum()
    if typeDataset == 'Test':
        golsPartidas=teams[teams['classificado_russia'] == 1].filter(items=['id']).rename(columns={'id': 'time'})
        golsPartidas['ano']=2018
        
    golsTrain = pd.DataFrame()

    golsTrain=golsPartidas
    golsTrain['goleiro']=0
    golsTrain['defesa']=0
    golsTrain['meio']=0
    golsTrain['ataque']=0

    for index, row in golsTrain.iterrows():
        time=row['time']
        defesa, ataque, goleiro, meio = desempenhoPosicao(time, teams)
        golsTrain.ix[(golsTrain['time'] == time), 'goleiro'] = goleiro
        golsTrain.ix[(golsTrain['time'] == time), 'defesa'] = defesa
        golsTrain.ix[(golsTrain['time'] == time), 'meio'] = meio
        golsTrain.ix[(golsTrain['time'] == time), 'ataque'] = ataque
    return golsTrain

#Funcao responsavel por pegar o desempenho de cada time por posicao
# da a cada time seu potencial defencivo, ofensivo bem como seu potencial criativo de meio campo
def desempenhoPosicao(team_id, teams):
    
    desempenho=['posicao', 'acceleration', 'aggression', 'agility', 'balance', 'ball_control', 'composure',
  'crossing', 'curve', 'dribbling', 'finishing', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes',
  'heading_accuracy', 'interceptions', 'jumping', 'long_passing', 'penalties', 'positioning']
  
    jogadoresTeam=jogadores[jogadores['team'] == getTeamName(team_id, teams)]
    jogadoresTeam=jogadoresTeam[list(desempenho)]
    media=jogadoresTeam.groupby(['posicao']).mean().mean(1)
       
    try:
        defesa = media['DF']
        ataque = media['FW']
        goleiro = media['GK']
        meio = media['MF']
    except KeyError:
        defesa = 0
        ataque = 0
        goleiro = 0
        meio = 0

    lista=[defesa, ataque, goleiro, meio]
    return lista

# Funcao responsavel por definir em colunas no proprio dataframe de treinamento 
# quem foi o ganhador de uma partida, perdedor ou se houve empate,
# quantos gols o vencedor, perdedor ou os times que empataram fez
# define tambem se time jogou em casa ou fora, neste caso, se o time esta do lado esquerdo ou direito da chave
def resultGame(train):
    conditions = [
    (train['gols_casa'] > train['gols_fora']), 
    train['gols_fora'] > train['gols_casa']]

    choices = [train['id_casa'], train['id_fora']]
    train['vencedor'] = np.select(conditions, choices, default=np.nan)
    train['vencedor'].fillna(-1, inplace=True) # atribuindo -1 caso for empate
    
    choices = [train['id_fora'], train['id_casa']]
    train['perdedor'] = np.select(conditions, choices, default=np.nan)
    train['perdedor'].fillna(-1, inplace=True) # atribuindo -1 casa for empate
    
    choices = [train['gols_casa'], train['gols_fora']]
    train['gols_vencedor'] = np.select(conditions, choices, default=np.nan)
    train['gols_vencedor'].fillna(train['gols_casa'], inplace=True)
    
    choices = [train['gols_fora'], train['gols_casa']]
    train['gols_perdedor'] = np.select(conditions, choices, default=np.nan)
    train['gols_perdedor'].fillna(train['gols_casa'], inplace=True)
    
    condition = [
                 (train['vencedor'] == train['id_casa']),
                  (train['vencedor'] == train['id_fora'])]
    
    choices = [1, -1]
    train['vencedor_onde'] = np.select(condition, choices, default=np.nan)
    train['vencedor_onde'].fillna(0, inplace=True)
    
# Funcao responsavel por gerar os insights de cada time, ou seja, a quantidade de vitorias que time teve,
# quantos titulos ela tem em copas, quantidade de derrotas, quantidade de participacoes em copa,
# total de jogos e suas respectivas medias, seu ranking na fifa, se um time e cabeca de chave e etc      
def getParticipacaoGeral(team_id):
    gamesWon = train[train.vencedor == team_id]
    totalGolsPartidaWon = gamesWon['gols_vencedor'].sum()
    gamesLost = train[train.perdedor == team_id]
    totalGolsPartidaLost = gamesLost['gols_perdedor'].sum()
    gamesDraw = train[(train.vencedor == -1) & ((train.id_casa == team_id) | (train.id_fora == team_id))]
    participacoes = teams[teams.id == team_id].values[0][6]
    totalGames = pd.concat([gamesWon, gamesDraw, gamesLost])
    numGames = len(totalGames.index)
    
    nVitorias = len(gamesWon.index)
    nDerrotas = len(gamesLost.index)
    nEmpate =  len(gamesDraw.index)
    
    if numGames == 0:
        mediaGolsMarcados=0
        mediaGolsSofridos=0
        mediaVitorias=0
        mediaDerrotas=0
        mediaEmpate=0
    else:
        mediaGolsMarcados=totalGolsPartidaWon/numGames
        mediaGolsSofridos=totalGolsPartidaLost/numGames
        mediaVitorias=nVitorias/numGames
        mediaDerrotas=nDerrotas/numGames
        mediaEmpate=nEmpate/numGames
    
    return [nVitorias, mediaGolsMarcados, mediaGolsSofridos, mediaVitorias, mediaDerrotas, mediaEmpate, participacoes, checkBestTeam(team_id, teams), getRankFifa(team_id, teams), checkCabecaChave(team_id, teams), desempenhoPosicao(team_id, teams)[0], desempenhoPosicao(team_id, teams)[1], desempenhoPosicao(team_id, teams)[2], desempenhoPosicao(team_id, teams)[3] ]

                     
# Gera uma "dicionarizacao" de todos os times em memoria
def createDic():
    #gerando uma lista de nomes dos times
    teamList = teams['nome'].tolist()
    dicFase = collections.defaultdict(list)
    for t in teamList:
        team_id = teams[teams['nome'] == t].values[0][0]
        team_vector = getParticipacaoGeral(team_id)
        dicFase[team_id] = team_vector
    return dicFase
   
# funcao ira criar toda a base de treinamento para a execucao do modelo de dados.  
def createTrainings():
    totalNumGames = len(train.index) #pega a quantidade de jogos historica de copas do mundo
    numFeatures= len(getParticipacaoGeral(447)) #desempenho do Brazil
    xTrain= np.zeros((totalNumGames, numFeatures + 1))
    yTrain= np.zeros((totalNumGames))
    
    team_dic = createDic() # faz a dicionarizacao dos dados
    
    count=0
    for index, row in train.iterrows():
        w_team = row['vencedor']
        w_vector = team_dic[w_team]
        l_team = row['perdedor']
        l_vector = team_dic[l_team]
        diff = [a - b for a, b in zip(w_vector, l_vector)]
        home = row['vencedor_onde']
        if (count % 2 == 0):
            diff.append(home)
            xTrain[count] = diff
            yTrain[count] = 1
        else:
            diff.append(-home)
            xTrain[count] = [ -p for p in diff]
            yTrain[count] = 0
        count +=1
    return xTrain, yTrain 

    
# Funcao cria efetivamente o modelo de predicao de acordo com a base de teste     
# lembrando que a base de teste e a tabele da copa da russia 2018
def createPrediction():
    results = [[0 for x in range(5)] for x in range(len(test.index))]
    for index, row in test.iterrows():
        team1_id = row['id_casa']
        team2_id = row['id_fora']
        team1_vector = getParticipacaoGeral(int(team1_id))
        team2_vector = getParticipacaoGeral(int(team2_id))
        pred = predictGame(team1_vector, team2_vector, 0, model)
        results[index][0] = row['partida_id']
        results[index][1] = getTeamName(team1_id, teams)
        results[index][2] = pred[0] * 100
        results[index][3] = getTeamName(team2_id, teams)
        results[index][4] = row['fase']

    results = pd.np.array(results)
    firstRow = [[0 for x in range(5)] for x in range(1)]
    firstRow[0][0] = 'Id'
    firstRow[0][1] = 'Time1'
    firstRow[0][2] = 'Porcentagem'
    firstRow[0][3] = 'Time2'
    firstRow[0][4] = 'Fase'
    with open("result.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(firstRow)
        writer.writerows(results)

# Funcao para fazer a predicao de desempenho entre dois times    
def predictGame(team_1_vector, team_2_vector, home, model):
    diff = [a - b for a, b in zip(team_1_vector, team_2_vector)]
    diff.append(home)
    return model.predict([diff])
    #return model.predict_proba([diff])
    
def prepareDatasetCards(dataset):
    features=['team', 'positioning', 'interceptions', 'heading_accuracy', 'ball_control', 'aggression']
    jogadoresCartoes=jogadores                    
    jogadoresCartoes=jogadoresCartoes[list(features)]
    media=jogadoresCartoes.groupby(['team'], as_index=False).mean() 
    
    
    dataset=dataset.join(media.set_index('team'), on='nome', how='left')
    
    featureTeam=['positioning', 'interceptions', 'heading_accuracy', 'ball_control', 'aggression']
    
    for f in featureTeam:
        dataset[f].fillna(0, inplace=True) #substituindo o valor NaN por 0
        
    return dataset
##########################################################################
######              Fim da definicao de funcoes
###########################################################################


##################################################################################################
######         Execucao do modelo de data mining no conjunto de dados ja preprocessados
##################################################################################################

# definindo os resultados das partidas para o conjunto de treinamento 
# ou seja, a funcao faz o pre-processamento da base de dados definindo o ganhador e perdedor de cada partida
resultGame(train) 

xTrain, yTrain = createTrainings() # executando a funcao de treinamento com base no dataframe train fase de grupos

# Dividindo o conjunto de dados em Treino e test e difinindo um percentual de teste
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(xTrain, yTrain, test_size=0.30, random_state = 0)

# definindo o modelo de treinamento 
# dando um fit no modelo passando o conjunto de teste e de treino  
# Foi executado diversos algoritmos e o linear_model.BayesianRidge() foi o que teve mais precisao para 
# a base de treinamento proposta
model = linear_model.BayesianRidge()
model.fit(X_train, Y_train)
preds = model.predict(X_test)

#Imprimindo a precisao do modelo
accuracy = metrics.accuracy_score(preds.round(),Y_test)
print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

# Criando a matriz de confusao
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, preds.round())

# criando a precisao do modelo em arquivo com base no conjunto de gerado de treino, teste e predicao  
createPrediction()

##################################################################################################
######      Fim da execucao do modelo de data mining no conjunto de dados ja preprocessados
##################################################################################################

#######################################################################################
############ Criando a tabela de classificacao com base no resultado da predicao
#######################################################################################
jogos=pd.read_csv('result.csv')

tabela=grupoTime[grupoTime['ano'] == 2018]
tabela=tabela.filter(items=['id', 'team', 'grupo'])
tabela['Pontos']=0
tabela['Vitorias']=0

for index, row in jogos.iterrows():    
    if row['Porcentagem'] >= 55:
        tabela.ix[(tabela.team == row['Time1']), 'Pontos'] += 3
        tabela.ix[(tabela.team == row['Time1']), 'Vitorias'] = getParticipacaoGeral(getTeamID(row['Time1'], teams))[0]

    if (row['Porcentagem'] < 55) & (row['Porcentagem'] >= 45):
        tabela.ix[(tabela.team == row['Time1']), 'Pontos'] += 1
        tabela.ix[(tabela.team == row['Time2']), 'Pontos'] += 1
        tabela.ix[(tabela.team == row['Time1']), 'Vitorias'] = getParticipacaoGeral(getTeamID(row['Time1'], teams))[0]
        tabela.ix[(tabela.team == row['Time2']), 'Vitorias'] = getParticipacaoGeral(getTeamID(row['Time2'], teams))[0]

    if row['Porcentagem'] < 45:
        tabela.ix[(tabela.team == row['Time2']), 'Pontos'] += 3
        tabela.ix[(tabela.team == row['Time2']), 'Vitorias'] = getParticipacaoGeral(getTeamID(row['Time2'], teams))[0]

tabela = tabela.sort_values(['grupo', 'Pontos', 'Vitorias'], ascending=[1, 0, 0])
tabela = tabela.reset_index(drop=True)

#######################################################################################
############ Trabalhando com a predicao para os times classificados para as oitavas
#######################################################################################

oitavas = []
#distribuindo os dois primeiros colocados em cada grupo para o chaveamento das oitavas
for i in range(0, 32, 8):
    if (i % 2 == 0):
        oitavas.append((int(tabela.loc[i][0]), int(tabela.loc[i+5][0]), i, 'oitavas'))
        oitavas.append((int(tabela.loc[i+1][0]), int(tabela.loc[i+4][0]), i+1, 'oitavas'))
oitavas = pd.DataFrame(oitavas)
oitavas=oitavas.rename(columns={0: 'id_casa', 1: 'id_fora', 2: 'partida_id', 3: 'fase'})

test=oitavas #jogando um uma nova base de test
createPrediction() #criando a predicao e jogando o resultado em arquivo

#######################################################################################
############  Definindo os ganhadores das oitavas e gerando o resultado para as quartas
#######################################################################################
vencedorOitavas = pd.read_csv('result.csv')
quartas = []
for i in range(0, 8, 2):
    if( vencedorOitavas.loc[i]['Porcentagem'] > 50.0):
        time1=vencedorOitavas.loc[i]['Time1']
    else:
        time1=vencedorOitavas.loc[i]['Time2']

    if( vencedorOitavas.loc[i+1]['Porcentagem'] > 50.0):
        time2=vencedorOitavas.loc[i+1]['Time1']
    else:
        time2=vencedorOitavas.loc[i+1]['Time2']

    quartas.append((int(getTeamID(time1, teams)), int(getTeamID(time2, teams)), i, 'quartas'))
quartas = pd.DataFrame(quartas)
quartas=quartas.rename(columns={0: 'id_casa', 1: 'id_fora', 2: 'partida_id', 3: 'fase'})

test=quartas #jogando um uma nova base de test
createPrediction() #criando a predicao e jogando o resultado em arquivo

#################################################################################################
############  Definindo os ganhadores das quartas e gerando o resultado para as semis-finais
#################################################################################################
vencedorQuartas = pd.read_csv('result.csv')
semi = []
for i in range(0, 4, 2):
    if( vencedorQuartas.loc[i]['Porcentagem'] > 50.0):
        time1=vencedorQuartas.loc[i]['Time1']
    else:
        time1=vencedorQuartas.loc[i]['Time2']

    if( vencedorQuartas.loc[i+1]['Porcentagem'] > 50.0):
        time2=vencedorQuartas.loc[i+1]['Time1']
    else:
        time2=vencedorQuartas.loc[i+1]['Time2']

    semi.append((int(getTeamID(time1, teams)), int(getTeamID(time2, teams)), i, 'semi'))
semi = pd.DataFrame(semi)
semi=semi.rename(columns={0: 'id_casa', 1: 'id_fora', 2: 'partida_id', 3: 'fase'})

test=semi #jogando um uma nova base de test
createPrediction() #criando a predicao e jogando o resultado em arquivo

#################################################################################################
############  Definindo os ganhadores das semi-finais e gerando o resultado para as finais
#################################################################################################
vencedorSemi = pd.read_csv('result.csv')
final = []
for i in range(0, 2, 2):
    if( vencedorSemi.loc[i]['Porcentagem'] > 50.0):
        time1=vencedorSemi.loc[i]['Time1']
    else:
        time1=vencedorSemi.loc[i]['Time2']

    if( vencedorSemi.loc[i+1]['Porcentagem'] > 50.0):
        time2=vencedorSemi.loc[i+1]['Time1']
    else:
        time2=vencedorSemi.loc[i+1]['Time2']

    final.append((int(getTeamID(time1, teams)), int(getTeamID(time2, teams)), i, 'final'))
final = pd.DataFrame(final)
final=final.rename(columns={0: 'id_casa', 1: 'id_fora', 2: 'partida_id', 3: 'fase'})

#define o confronto do terceiro e quarto lugar
terceiroQuarto = []
for i in range(0, 2, 2):
    if( vencedorSemi.loc[i]['Porcentagem'] < 50.0):
        time1=vencedorSemi.loc[i]['Time1']
    else:
        time1=vencedorSemi.loc[i]['Time2']

    if( vencedorSemi.loc[i+1]['Porcentagem'] < 50.0):
        time2=vencedorSemi.loc[i+1]['Time1']
    else:
        time2=vencedorSemi.loc[i+1]['Time2']

    terceiroQuarto.append((int(getTeamID(time1, teams)), int(getTeamID(time2, teams)), i, 'terceiro'))
terceiroQuarto = pd.DataFrame(terceiroQuarto)
terceiroQuarto=terceiroQuarto.rename(columns={0: 'id_casa', 1: 'id_fora', 2: 'partida_id', 3: 'fase'})

test=terceiroQuarto
createPrediction()

vencedorTerceiroQuarto = pd.read_csv('result.csv')


test=final #jogando um uma nova base de test
createPrediction() #criando a predicao e jogando o resultado em arquivo

vencedorFinal = pd.read_csv('result.csv')

#Variavel contem os resultados finais das predicoes
resultadosFinais=pd.concat([vencedorOitavas, vencedorQuartas, vencedorSemi, vencedorFinal]) #criando um consolidado de resultados finais


print('*****    Resultados Finais dos jogos   ******')
print(resultadosFinais)

def resultadosFinais(dataframe):
    if dataframe['Porcentagem'].values > 50:
        ganhador=dataframe['Time1'][0]
        perdedor=dataframe['Time2'][0]
    else:
        ganhador=dataframe['Time2'][0]
        perdedor=dataframe['Time1'][0]

    return [ganhador, perdedor]

##############################################################################################
# Define que serah o time que mais ira fazer gols na copa
# Como parametro de analise serah criado uma base com potencial de cada time, bem como
# seu historico de gols em copas passadas
#############################################################################################                 

golsTrain = createGols(train, 'Train')
golsTest = createGols(teams, 'Test')
fieldsTrain=['time', 'ano', 'goleiro', 'defesa', 'meio', 'ataque']
target=['gols']
xTrainGols=golsTrain[list(fieldsTrain)].values
yTrainGols=golsTrain[target].values
xTestGols=golsTest[list(fieldsTrain)].values
y = yTrainGols.ravel()
yTrainGols = np.array(y).astype(int)

from sklearn.cross_validation import train_test_split
X_Golstrain, X_Golstest, y_Golstrain, y_test = train_test_split(xTrainGols, yTrainGols, test_size=0.30, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, max_features=4)

classifier.fit(X_Golstrain, y_Golstrain)

# Predicao dos resultados da base de teste
y_pred = classifier.predict(X_Golstest)

# Predicao da base de times que estao na copa da Russia
y_pred = classifier.predict(xTestGols)

gols=pd.DataFrame(y_pred).rename(columns={0: 'Qtde'})
xTestGols=pd.DataFrame(xTestGols)
golsSelecoes=pd.concat([xTestGols, gols], axis=1).rename(columns={0: 'time'})

maiorGoleador=int(golsSelecoes.ix[golsSelecoes['Qtde'].idxmax()][0])

print('Time goleador da Copa Russia 2018: ' + getTeamName(maiorGoleador, teams))

#####################################################################################
#       Prepara a base e monta o modelo preditivo para Cartoes
#       O modelo ira responder mais pais com mais cartoes amarelos e vermelhos
#####################################################################################

#coloca na base de cartoes informacoes do tipo agressividade, posicionamento, controle de bola
cartoes=prepareDatasetCards(cartoes)
cartoesApp=cartoes.drop(['nome'], axis=1)

target=['qtde']

features=list(set(list(cartoesApp.columns))-set(target))
xCartoesTrain=cartoesApp[list(features)].values
yCartoesTrain=cartoesApp[target].values

#carrega a base de test para a predicao do modelo de cartoes
xCartoesTest=prepareDatasetCards(cartoesTest)
xCartoesTest=xCartoesTest.drop(['nome'], axis=1).values

X_Cartoestrain, X_Cartoestest, y_Cartoestrain, y_test = train_test_split(xCartoesTrain, yCartoesTrain, test_size=0.30, random_state = 0)

#classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0, max_features=4)    

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_Cartoestrain, y_Cartoestrain)

# Predicao dos resultados da base de teste
y_pred = classifier.predict(X_Cartoestest)

# Predicao dos resultados da base de teste
y_pred = classifier.predict(xCartoesTest)

cartoesQuantidade=pd.DataFrame(y_pred).rename(columns={0: 'Qtde'})
xCartoesTest=pd.DataFrame(xCartoesTest)
predictCartoes=pd.concat([xCartoesTest, cartoesQuantidade], axis=1).rename(columns={0: 'time'})
maisCartoes=predictCartoes.groupby(['time'], as_index=False).Qtde.sum()
maisCartoes=int(maisCartoes.ix[maisCartoes['Qtde'].idxmax()][0])

print('Pais com mais cartoes amarelos e vermelhos: ' + getTeamName(maisCartoes, teams))


print('**** Quarto Lugar: ' + resultadosFinais(vencedorTerceiroQuarto)[1])
print('***  Terceiro Lugar: ' + resultadosFinais(vencedorTerceiroQuarto)[0])
print('**   Vice-Campeao: ' + resultadosFinais(vencedorFinal)[1])
print('*    Campeao: ' + resultadosFinais(vencedorFinal)[0])