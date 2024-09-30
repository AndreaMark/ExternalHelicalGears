# SpurExternalGears

Questa cartella contiene tutto il necessario ad un dimensionamento di massima e ad una verifica di ruote dentate cilindriche a denti elicoidali, esterni. 

All'interno della cartella "moduli" sono contenuti tutte le funzioni necessarie al calcolo dei fattori geometrici e di resistenza. 
  - ruote_fattorigeometrici
    Contiene tutte le funzioni necessarie al primo dimensionamento della ruota dentata: numero di denti, grandezze caratteristiche, strisciamenti...
  - ruote_correzione
    Contiene una funzione iterativa necessaria al calcolo dei coefficienti di correzione e delle caratteristiche corrette della ruota dentata.
  - ruote_fattoriinfluenza
    Contiene tutte quelle funzioni necessarie al calolo dei fattori di influenza secondo norma ISO 6336-1
  - ruote_fattoriflessione
    Contiene la maggiore parte delle funzioni necessarie al calcolo dei coefficienti correttivi di flessione secondo norma ISO 6336-3
  - ruote_fattoripitting
    Contiene la maggior parte delle funzione necessarie al calcolo dei coefficienti correttivi di pressione da contatto secondo norma ISO 6336-2
  - ruote_fattorigrippaggio
    Contiene la maggior parte delle funzioni necessario al calcolo dei coefficiente e delle temperature di esercizio all'interno di un riduttore secondo norma ISO 6336-20 (Vullo-Gears-Vol2)
  - ruote_interpolazione
    Contiene una funzione utile al calcolo dell'iterpolazione lineare di c oefficienti in tabella.

L'ordine in cui runnare gli script è chiaramente il seguente: 
1 - 2 - 3.1 - 3.2 - 4

Ogni script creerà un database dal quale lo script successivo attingerà valori, una tabella al fine di visualizzare i valori, un file "log" dove registererà i print e delle immagini. 
Ogni file viene salvato nella sua relativa cartella, che per semplicità esecutiva vengono salvati all'interno della stessa cartella dello script. 

Nella cartella "ipotesi preliminari" ci sono tre tabelle excel che si consglia di compilare prima di runnare qualsiasi script. 

Sarà necessario inserire in input - non appena richiesto dallo script - soltanto alcuni valori all'interno del file 1, come: 
- Il numero di giri in ingresso in [rpm]
- Il rapporto di riduzione velocità scelto
- Il numero di denti del pignone
- Il valore del modulo unificato scelto
- ATTENZIONE: prima di runnare lo script è necessario modificare i valori alle righe 192, 193 altrimenti si ottengono de valori di Lewis non corretti per i numero di denti scelto.

Il file "4" produce solo un log e delle immagini, nè tabelle nè database

Andrea Marchegiani 
18/10/2023

Il file "forma_dente_jup" è un file jupiter dal quale poter estrapolare due grafici principali, quelli della forma del dente costruito dalla fondamentale non corretto e corretto.
Non è necessario dover inserire valori poiché li attinge automaticamente dai database, se la dentatura non è corretta produrrà ovviemente due file uguali.

Andrea Marchegiani
17/12/2023
