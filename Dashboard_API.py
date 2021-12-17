import streamlit as st
import pandas as pd
import numpy as np
#import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image
import joblib
warnings.simplefilter(action='ignore', category=FutureWarning)

st.title('Application Modèle scoring, crédit de consommation !!!')

 
img = Image.open("Pret_a_depenser.PNG") 
  
st.image(img, width=200)

#-------------------------------------------------------
#Charger les données

st.subheader('Aperçu data')
st.text('Loading data...')
df = pd.read_csv('bd_equi.csv').drop(columns='Unnamed: 0')
st.write(df)

#Rechercher par id_client
Id = st.text_input("Saisir l'Id du client :")
if (st.button('Validez')):
	st.write(df[df.SK_ID_CURR == int(Id)])

#-------------------------------------------------------
#Définition fonction Tableau de board
def dashboard(df):
	fig = plt.figure(1)

	plt.subplot(121)
	df['TARGET'].value_counts(normalize=True).plot.bar(title='Crédit accordé ou pas')

	plt.subplot(122)
	df['CNT_CHILDREN'].value_counts(normalize=True).plot.bar(title="Répartition selon le nombre d'enfant")

	plt.show()
	st.pyplot(fig)
	#--------------------------------------------------------
	fig1 = plt.figure(1)
	plt.subplot(121)
	df['CODE_GENDER'].value_counts(normalize=True).plot.bar(title="Répartition selon le genre")

	#sns.countplot(x='CODE_GENDER',data=df)
	#ax2.title('Genre du client')

	plt.subplot(122)
	df['FLAG_OWN_REALTY'].value_counts(normalize=True).plot.bar(title="Propriétaire de logement ou non")

	plt.show()
	st.pyplot(fig1)

dashboard(df)
#----------------------------------------------------------
# Distribution Montant total de revenu
fig3 = plt.figure(2)
plt.subplot(121)
sns.distplot(df['AMT_INCOME_TOTAL'].apply(np.log))

plt.subplot(122)
sns.distplot(df['AMT_ANNUITY'].apply(np.log))
st.pyplot(fig3)


#-------------------------------------------------------
#Collecter l'entrée
st.sidebar.header("Visualiser avec un filtre")
filtre = st.sidebar.selectbox('Choisir le mode de filtre :', ('genre_client', 'Nbre_enfant', 'ID'))
#agree = st.button('Cliquez')
if (filtre == 'genre_client') :
	Genre = st.sidebar.selectbox('Genre client',(df.CODE_GENDER.unique()))
	if Genre == 0:
		st.write('Vous avez choisis de filtrer selon le genre feminin !')
	else : st.write('Vous avez choisis de filtrer selon le genre masculin !')
	st.write(df[df.CODE_GENDER == Genre],
		'Le revenu moyende ce groupe est de ',
		df[df.CODE_GENDER == Genre].AMT_INCOME_TOTAL.mean())

	dashboard(df[df.CODE_GENDER == Genre])

elif (filtre == 'Nbre_enfant') :
	Nbre_enfant = st.sidebar.selectbox('Nombre d''enfant',(df.CNT_CHILDREN.unique()))
	st.write(df[df.CNT_CHILDREN == Nbre_enfant],
		'Le revenu moyende ce groupe est de ',
		df[df.CNT_CHILDREN == Nbre_enfant].AMT_INCOME_TOTAL.mean())

	dashboard(df[df.CNT_CHILDREN == Nbre_enfant])

elif (filtre == 'ID') :
	ID = st.sidebar.selectbox('Iditifiant client',(df.SK_ID_CURR.to_list()))
	st.write(df[df.SK_ID_CURR == ID])

	dashboard(df[df.SK_ID_CURR == ID])

else :
	st.write('Choisissez le filtre que vous souhaitez')

#-------------------------------------------------------

#Collecter le profil d'entrée
st.sidebar.header("Les caracteristiques du client")

def client_caract_entree():
    Gender=st.sidebar.selectbox('Sexe',('0','1'))
    Nb_enf=st.sidebar.selectbox('Nombre Enfants',('0','1','2','3'))
    Vehicule=st.sidebar.selectbox('Véhiculé',('0','1'))
    Propriétaire_logement=st.sidebar.selectbox('Propriétaire_logement',('0','1'))
    Montant_revenu=st.sidebar.slider('Montant des revenu du client',150,1000000,45000)
    #Montant_revenu = float(st.sidebar.text_input("Saisir le Montant_revenu :",0))
    Montant_Credit=st.sidebar.slider('Montant du crédit en dollar',1000,500000,2500)
    #Montant_Credit = float(st.sidebar.text_input("Saisir le Montant_Credit :" ,0))
    DAYS_BIRTH=st.sidebar.slider('DAYS_BIRTH',-150,13000,200)
    #DAYS_BIRTH = float(st.sidebar.text_input("Saisir DAYS_BIRTH :",0))
    Montant_Rente=st.sidebar.slider('Montant de la rente',150,30000,1000)
    #Montant_Rente = float(st.sidebar.text_input("Saisir Montant_Rente :",0))

    data={
    'Gender':Gender,
    'Nb_enf':Nb_enf,
    'Vehicule':Vehicule,
    'Propriétaire_logement':Propriétaire_logement,
    'Montant_revenu':Montant_revenu,
    'Montant_Credit':Montant_Credit,
    'DAYS_BIRTH':DAYS_BIRTH,
    'Montant_Rente':Montant_Rente
    }

    profil_client=pd.DataFrame(data,index=[0])
    return profil_client

input_df=client_caract_entree()

#st.write(input_df)

#importer le modèle
#load_model=pickle.load(open('xgb_cl_prevision_credit.pkl','rb'))

load_model = joblib.load('xgb_cl_prevision_credit.joblib')


#appliquer le modèle sur le profil d'entrée
agree = st.sidebar.button('Validez', 0)
if (agree):
    prevision=load_model.predict(input_df)

    st.subheader('Résultat de la prévision')
    if int(prevision) == 0:
        st.write("Classe '0'")
    else :
        st.write("Classe '1'")


#--------------------------------------------------------------------

