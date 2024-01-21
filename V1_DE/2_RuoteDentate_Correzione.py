import numpy as np
import matplotlib.pyplot as plt
from moduli import ruote_fattorigeometrici as rfge
from moduli import ruote_fattoripitting as rfp
from moduli import ruote_correzione as rc
import pandas as pd
from tabulate import tabulate
import sqlite3
import os
import shutil
from datetime import datetime
####    NB - La correzione avviene sul piano NORMALE -> RUOTE FITTIZIE
################################################            CORPO            ################################################  
start_time = datetime.now()

out_file = open("Correzione.txt", "w")                                # scrivo il file log

######################################################    Importazione Valori dai database
####    VALORI GLOBALI
vg_path = "database/variabili_globali.db"
# Connessione
conn = sqlite3.connect(vg_path)
cur = conn.cursor()
cur.execute("SELECT valori FROM variabili_globali")
vg = [row[0] for row in cur.fetchall()]
theta_p = vg[0]                                       # Angolo di pressione normale
theta_f = vg[6]                                       # Angolo di pressione frontale
DentiP = vg[1]
DentiR = vg[2]                                        # Numero di denti di pignone e ruota
tau_re = DentiP/DentiR                                # Reale rapporto di trasmissione
m = vg[4]                                             # modulo e addendum
a = m
beta = vg[5]                                          # angolo d'elica
beta_f = vg[10]                                       # angolo d'elica fondamentale
b =  vg[9]                                            # larghezza di fascia
conn.close()





####    CARATTERISTICHE GEOMETRICHE NON CORRETTE - piano frontale
cnc_path = "database/caratteristiche_nc.db"
conn = sqlite3.connect(cnc_path)
#   PIGNONE
cur1 = conn.cursor()
cur1.execute("SELECT pignone FROM caratteristiche_nc")
cncP = [row[0] for row in cur1.fetchall()]
#   RUOTA
cur2 = conn.cursor()
cur2.execute("SELECT ruota FROM caratteristiche_nc")
cncR = [row[0] for row in cur2.fetchall()]
####    Pignone
r_pignone = cncP[0]     # Primitiva
rho_pignone = cncP[1]   # Fondamentale
####    Ruota
r_ruota = cncR[0]       # Primitiva
rho_ruota = cncR[1]     # Fondamentale
conn.close()





####    VARIABILI CINEMATICHE
vc_path = "database/variabili_cinematiche.db"
conn = sqlite3.connect(vc_path)
#   PIGNONE
cur1 = conn.cursor()
cur1.execute("SELECT pignone FROM variabili_cinematiche")
vcP = [row[0] for row in cur1.fetchall()]
#   RUOTA
cur2 = conn.cursor()
cur2.execute("SELECT ruota FROM variabili_cinematiche")
vcR = [row[0] for row in cur2.fetchall()]

Momento_pignone = vcP[1]                        # Nm
Momento_ruota = vcR[1]                          # Nm
conn.close()





####    Raggi di testa di ruota e pignone NON CORRETTI - FRONTALI
Rtp = r_pignone + a
Rtr = r_ruota + a

####    Calcolo del segmento dei contatti NON CORRETTO - FRONTALI
phi = np.pi/2 - theta_p
accesso_reale = np.sqrt(Rtr**2 - rho_ruota**2) - r_ruota*np.cos(phi)
recesso_reale = np.sqrt(Rtp**2 - rho_pignone**2) - r_pignone*np.cos(phi)
delta = np.arange(-accesso_reale, recesso_reale, 0.01)





####    Correzione dentatura 
####    x, x' si ricavano SEMPRE dalle ruote fittizie, poi all'interno dello scopri procedo a modificarle per il piano frontale
correzione = rc.correzione_dentature(tau_re, delta, theta_f, 0.31, 0.2, rho_pignone, rho_ruota, r_pignone, r_ruota, DentiP, DentiR, m, beta, theta_p)
st97 = f"Il numero di iterazioni è: {correzione[14]}"
out_file.write(st97 + '\n'), print(st97)



####    SEGMENTO DEI CONTATTI CORRETTO E PLOT
delta_new = np.arange(-correzione[3], correzione[4], 0.01)

ks_pignone_plot2 = [rfge.ks_pignone(tau_re, dd, r_pignone, correzione[2]) for dd in delta_new]
ks_ruota_plot2 = [rfge.ks_ruota(tau_re, dd, r_ruota, correzione[2]) for dd in delta_new]

x_pignone_approssimato = round(correzione[0], 2)
x_ruota_approssimato = round(correzione[1], 2)
titolo = 'Grafico degli strisciamenti specifici (x={}  x\'={})'.format(x_pignone_approssimato, x_ruota_approssimato)
plt.figure(num=3)
plt.plot(delta_new, ks_pignone_plot2, "b", label = "pignone")
plt.plot(delta_new, ks_ruota_plot2, "r", label = "ruota")
plt.xlabel('Segmento dei contatti [mm]')
plt.ylabel(r'$k_s$')
plt.title(titolo)
plt.legend()
plt.grid()
plt.savefig('strisciamenti_c.png', dpi=300)

ks_max_pignone_corretto = max(np.abs(np.min(ks_pignone_plot2)), np.abs(np.max(ks_pignone_plot2)))
ks_max_ruota_corretto = max(np.abs(np.min(ks_ruota_plot2)), np.abs(np.max(ks_ruota_plot2)))
delta_ks_corretto = np.abs(ks_max_pignone_corretto - ks_max_ruota_corretto)

tab2 = [ ["ksmax", round(ks_max_pignone_corretto, 3), round(ks_max_ruota_corretto,3)],
        ["delta k", round(delta_ks_corretto, 3), round(delta_ks_corretto, 3) ]]

st1 = "Le verifiche sullo strisciamento sono superate!"
st2 = tabulate(tab2, headers=["Pignone", "Ruota"])
out_file.write(st1 + '\n'), print(st1)
out_file.write(st2 + '\n'), print(st2)





####    Verifiche corrette - SUL PIANO FRONTALE 
####    Spessore della testa del dente corretto
evgamma_corr_Pignone = rfge.ev(theta_p) + (2/DentiP)*(np.pi/4 + x_pignone_approssimato*np.tan(theta_p)) 
evgamma_corr_Ruota = rfge.ev(theta_p) + (2/DentiR)*(np.pi/4 + x_ruota_approssimato*np.tan(theta_p))

####    Angolo di pressione in testa
theta_t_corr_pignone = np.arccos(rho_pignone/(correzione[5]))
theta_t_corr_ruota = np.arccos(rho_ruota/(correzione[6]))

spessore_testa_corretto_pignone = 2*(correzione[5])*(evgamma_corr_Pignone - rfge.ev(theta_t_corr_pignone))
spessore_testa_corretto_ruota = 2*(correzione[6])*(evgamma_corr_Ruota - rfge.ev(theta_t_corr_ruota))

spessore_lim = 0.2*correzione[7]

tab_spesst = [["ev_gamma [rad]", round(evgamma_corr_Pignone, 4), round(evgamma_corr_Ruota, 4)], 
              ["Angolo in testa [rad]", round(theta_t_corr_pignone, 4), round(theta_t_corr_ruota, 4)],
              ["Spessore testa [mm]", round(spessore_testa_corretto_pignone, 2), round(spessore_testa_corretto_ruota, 2)],
              ["Spessore limite [mm]", spessore_lim, spessore_lim]]

st3 = "La verifica sullo spessore è superata!"
st4 = "Ritenta, spessore in testa troppo basso!"
st5 = tabulate(tab_spesst, headers=["Pignone", "Ruota"])

if spessore_testa_corretto_pignone>spessore_lim and spessore_testa_corretto_ruota>spessore_lim :
    out_file.write(st3 + '\n'), print(st3)
else:
    out_file.write(st4 + '\n'), exit(st4)

out_file.write(st5 + '\n'), print(st5)





####    Interferenza
phi_new = np.pi/2 - correzione[2]

t1c = r_pignone*np.cos(phi_new)     # Accesso Teorico

ct2 = r_ruota*np.cos(phi_new)       # Recesso Teorico

accesso_reale_new = correzione[3]

recesso_reale_new = correzione[4]

tab1 = [ ["Teorico", round(t1c,3), round(ct2,3)],
     ["Reale", round(accesso_reale_new,3), round(recesso_reale_new,3)]]

st6 = f"La verifica sull'interferenza è superata {round((t1c + ct2),2)}, {round((accesso_reale_new + recesso_reale_new),2)}"
st7 = "C'è qualcosa che non va!"
st8 = tabulate(tab1, headers=["Accesso", "Recesso"])

if accesso_reale_new < t1c and recesso_reale_new < ct2:
    out_file.write(st6 + '\n'), print(st6)
else:
    out_file.write(st7 + '\n'), exit(st7)

out_file.write(st8 + '\n'), print(st8)





####    Fattore di ricoprimento
contatti_new = accesso_reale_new + recesso_reale_new

passo_new = correzione[7]*np.pi

epsilon_verifica_ev_new = contatti_new/(passo_new*np.cos(correzione[2]))
epsilon_verifica_el_new = (30*np.tan(beta))/passo_new

epsilon_verifica_new =  epsilon_verifica_ev_new +  epsilon_verifica_el_new 

st9 = f"Il fattore di ricoprimento è {round(epsilon_verifica_new, 1), [round(epsilon_verifica_ev_new, 1), round(epsilon_verifica_el_new, 1)]}"
st10 = "Il fattore di ricoprimento non è abbastanza!"

if round(epsilon_verifica_new, 1)>=1.2:
    out_file.write(st9 + '\n'), print(st9)
else:
    out_file.write(st10 + '\n'), exit(st10)





####    Grandezze caratteristiche
####    Spessori primitive di taglio
s_taglio_pignone = 2*r_pignone*(evgamma_corr_Pignone - rfge.ev(theta_p))
s_taglio_ruota = 2*r_ruota*(evgamma_corr_Ruota - rfge.ev(theta_p))

####    Spessori primitive di lavoro
s_lavoro_pignone = 2*correzione[8]*(evgamma_corr_Pignone - rfge.ev(correzione[2]))
s_lavoro_ruota = 2*correzione[9]*(evgamma_corr_Ruota - rfge.ev(correzione[2]))





####    Altre grandezze necessarie
addendum_pignone = round(correzione[5]-correzione[8], 2)
addendum_ruota = round(correzione[6]-correzione[9], 2)
dedendum_pignone = round(correzione[8]-correzione[10], 2)
dedendum_ruota = round(correzione[9]-correzione[11], 2)
interasse = round(correzione[8] + correzione[9], 2)
deltad = (abs(r_pignone-correzione[8]) + abs(r_ruota-correzione[9]))
gioco_radiale = 2*deltad*np.sin(correzione[2])

alpha_t = np.arctan(np.tan(theta_p)/np.cos(beta))                                           # angolo di pressione trasversale di riferimento UNI 8862 tab4

alpha_wt = np.arccos(((2*r_pignone)*np.cos(alpha_t))/(2*correzione[8]))                     # angolo di pressione trasversale di funzionamento UNI8862 tab4

rho_red = rfp.curvatura_relativa((2*correzione[10]), (2*round(correzione[11])), alpha_wt)   # curvatura relativa





####    TabellE riassuntivE
xnf = x_pignone_approssimato*np.cos(beta)
xnf = x_ruota_approssimato*np.cos(beta)

#   FRONTALE
tab_grandezze = [ ["Coefficienti di correzione", round(xnf,2), round(xnf,2)],
                    ["Angolo di lavoro frontale [rad]", round(correzione[2], 5), round(correzione[2],5)], 
                    ["Angolo di lavoro frontale [grad]", round(np.degrees(correzione[2]), 2), round(np.degrees(correzione[2]), 2)],
                    ["Modulo frontale [mm]", round(correzione[7],3), round(correzione[7],3)],
                    ["Addendum [mm]", addendum_pignone, addendum_ruota],
                    ["Dedendum [mm]", dedendum_pignone, dedendum_ruota],
                    ["Raggio primitiva di lavoro [mm]", round(correzione[8], 2), round(correzione[9], 2)],
                    ["Raggio di testa [mm]", round(correzione[5],2), round(correzione[6],2)],
                    ["Raggio di piede [mm]", round(correzione[10],2), round(correzione[11],2)],
                    ["Raggio Fondamentale [mm]", round(rho_pignone,2), round(rho_ruota,2)],
                    ["Distanze tra primitive di lavoro e di taglio", round(correzione[12], 3), round(correzione[13], 3)], 
                    ["Larghezza di fascia [mm]", b, "="],
                    ["Spessore su primitiva di taglio [mm]", round(s_taglio_pignone,2), round(s_taglio_ruota,2)],
                    ["Spessore su primitiva di lavoro [mm]", round(s_lavoro_pignone,2), round(s_lavoro_ruota,2)],
                    ["Interasse [mm]", interasse, "="],
                    ["Gioco radiale [mm]", round(gioco_radiale, 2), "="],
                    ["Angolo di pressione trasversale di riferimento [rad]", round(alpha_t, 5), round(alpha_t, 5)],
                    ["Angolo di pressione trasversale di funzionamento [rad]",round(alpha_wt, 5), round(alpha_wt, 5) ],
                    ["Curvatura relativa", round(rho_red, 2), round(rho_red, 2)], 
                    ["Accesso", round(accesso_reale_new,2), "\\"],
                    ["Recesso", "\\", round(recesso_reale_new,2)],
                    ["Fattore di ricoprimento d'evolvente", round(epsilon_verifica_ev_new, 1), round(epsilon_verifica_ev_new, 1)],]

st11 = "A queste ruote si associano le seguenti grandezze caratteristiche CORRETTE FRONTALI"
st12 = tabulate(tab_grandezze, headers=["Pignone", "Ruota"])
out_file.write(st11 + '\n'), print(st11)
out_file.write(st12 + '\n'), print(st12)

#   NORMALE


theta_ln = theta_lavoro_n = np.arctan(np.tan(correzione[2]*np.cos(beta)))

mn = correzione[7]*np.cos(beta)
an_p = m
un_p = 1.25*m

z_np = int(DentiP/np.cos(beta)**3)+1
z_nr = int(DentiR/np.cos(beta)**3)+1

##   Raggi primitivi
r_pignone_n = rfge.raggi_primitive(mn, z_np)
r_ruota_n = rfge.raggi_primitive(mn, z_nr)

##   Raggio di testa
Rtp_n = r_pignone_n + addendum_pignone
Rtr_n = r_ruota_n + addendum_ruota

##   Raggio di piede
Rpp_n = r_pignone_n - dedendum_pignone
Rpr_n = r_ruota_n - dedendum_ruota

##   Raggio fondamentale
rho_pignone_n = r_pignone_n*np.cos(theta_ln)
rho_ruota_n = r_ruota_n*np.cos(theta_ln)

tab_grandezze2 = [ ["Coefficienti di correzione", x_pignone_approssimato, x_ruota_approssimato],
                    ["Angolo di lavoro normale [rad]", round(theta_ln,5), "="], 
                    ["Angolo di lavoro normale [grad]", np.degrees(theta_ln), "="],
                    ["Modulo normale [mm]", round(mn,3), round(mn,3)],
                    ["Addendum [mm]", addendum_pignone, addendum_ruota],
                    ["Dedendum [mm]", dedendum_pignone, dedendum_ruota],
                    ["Raggio primitiva di lavoro [mm]", round(r_pignone_n, 2), round(r_ruota_n, 2)],
                    ["Raggio di testa [mm]", round(Rtp_n,2), round(Rtr_n,2)],
                    ["Raggio di piede [mm]", round(Rpp_n,2), round(Rpr_n,2)],
                    ["Raggio Fondamentale [mm]", round(rho_pignone_n,2), round(rho_ruota_n,2)],]

st99 = "A queste ruote si associano le seguenti grandezze caratteristiche CORRETTE NORMALI"
st98 = tabulate(tab_grandezze2, headers=["Pignone", "Ruota"])
out_file.write(st99 + '\n'), print(st99)
out_file.write(st98 + '\n'), print(st98)


####    Forze agenti
raggio_pignone_forza = correzione[8]*0.001                      # m
raggio_ruota_forza = correzione[9]*0.001                        # m

F_tangenziale_pignone = Momento_pignone/raggio_pignone_forza
F_tangenziale_ruota = Momento_ruota/raggio_ruota_forza

F_assiale_pignone = F_tangenziale_pignone*(np.tan(beta_f)/np.cos(correzione[2]))
F_assiale_ruota = F_tangenziale_ruota*(np.tan(beta_f)/np.cos(correzione[2]))

F_radiale_pignone = F_tangenziale_pignone*np.tan(correzione[2])
F_radiale_ruota = F_tangenziale_ruota*np.tan(correzione[2])

tab_forze = [["Forze circonferenziali [N]", round(F_tangenziale_pignone, 2), round(F_tangenziale_ruota, 2)],
             ["Forze radiali [N]", round(F_radiale_pignone, 2), round(F_radiale_ruota, 2) ], 
             ["Forze assiali [N]", round(F_assiale_pignone,2), round(F_assiale_ruota,2)]]

st13 = "Le forze agenti su queste ruote sono:"
st14 = tabulate(tab_forze, headers=["Pignone", "Ruota"])
out_file.write(st13 + '\n'), print(st13)
out_file.write(st14 + '\n'), print(st14)





################################################            DATABASE        ################################################
out_file.close()                                                        # chiudo e salvo il file log

post_correzione = {
    'variabili':["Coefficienti di correzione",
                 "Angolo di lavoro [rad]",
                 "Angolo di lavoro [grad]",
                 "Modulo [mm]",
                 "Addendum [mm]",
                 "Dedendum [mm]",
                 "Raggio primitiva di lavoro [mm]",
                 "Raggio di testa [mm]",
                 "Raggio di piede [mm]",
                 "Raggio Fondamentale [mm]",
                 "Distanze tra primitive di lavoro e di taglio",
                 "Larghezza di fascia [mm]",
                 "Spessore su primitiva di taglio [mm]",
                 "Spessore su primitiva di lavoro [mm]",
                 "Interasse [mm]",
                 "Gioco radiale [mm]", 
                 "Fattore di ricoprimento",
                 "Forze circonferenziali [N]",
                 "Forze radiali [N]",
                 "Angolo di pressione trasversale di riferimento [rad]",
                 "Angolo di pressione trasversale di funzionamento [rad]",
                 "Angolo in testa [rad]", 
                 "Curvatura relativa [mm]", 
                 "Accesso",
                 "Recesso",                             
                 "Fattore di ricoprimento d'evolvente",
                 "Diametro di testa NORMALE",
                 ],
    'pignone':[x_pignone_approssimato,
               round(correzione[2], 5),
               round(np.degrees(correzione[2]), 2),           
               round(correzione[7],3),
               addendum_pignone,
               dedendum_pignone,
               round(correzione[8], 2),
               round(correzione[5],2),
               round(correzione[10],2),
               round(rho_pignone,2),
               round(correzione[12], 3),
               b,
               round(s_taglio_pignone,2),
               round(s_lavoro_pignone,2),
               interasse,
               round(gioco_radiale, 2),
               round(epsilon_verifica_new, 1),
               round(F_tangenziale_pignone, 2),
               round(F_radiale_pignone, 2),
               round(alpha_t, 5),
               round(alpha_wt, 5),
               round(theta_t_corr_pignone,5), 
               round(rho_red, 2), 
               round(accesso_reale_new,2),
               0,
               round(epsilon_verifica_ev_new, 1),
               round(Rtp_n,2), ],
    'ruota':[x_ruota_approssimato,
             round(correzione[2],5),
             round(np.degrees(correzione[2]), 2),            
             round(correzione[7],3),
             addendum_ruota,
             dedendum_ruota,
             round(correzione[9], 2),
             round(correzione[6],2),
             round(correzione[11],2),
             round(rho_ruota,2),
             round(correzione[13], 3),
             25,
             round(s_taglio_ruota,2),
             round(s_lavoro_ruota,2),
             interasse,
             round(gioco_radiale, 2),
             round(epsilon_verifica_new, 1),
             round(F_tangenziale_ruota, 2),
             round(F_radiale_ruota, 2),
             round(alpha_t, 5),
             round(alpha_wt, 5),
             round(theta_t_corr_ruota,5),
             round(rho_red, 2), 
             0,
             round(recesso_reale_new,2),
             round(epsilon_verifica_ev_new, 1),
             round(Rtr_n,2)],             
}
index_labels1 = ['r1', 'r2', 'r3', 'r4', 'r5', 
                 'r6', 'r7', 'r8', 'r9', 'r10', 
                 'r11', 'r12', 'r13', 'r14', 'r15', 
                 'r16', 'r17', 'r18', 'r19', 'r20',
                 'r21', 'r22', 'r23', 'r24', 'r25', 'r26', 'r27']

PC = pd.DataFrame(post_correzione, index=index_labels1)

# Connessione al database SQLite
conn = sqlite3.connect("post_correzione.db")

# Salva il DataFrame nel database
PC.to_sql('post_correzione', conn, if_exists='replace', index_label='index')

# Chiudi la connessione
conn.close()

# Tabella su excel sempre facilmente consultabile
PC.to_excel('correzione.xlsx', index=True)









################################################            ORGANIZZAZIONE FILE        ################################################
# Ottieni il percorso della directory in cui si trova lo script
percorso_script = os.path.dirname(os.path.abspath(__file__))

# Costruisci il percorso completo della cartella di origine
cartella_origine = os.path.join(percorso_script)

# Crea una cartella per le figure, tabelle e log all'interno della directory dello script (senza errori se esistono già)
os.makedirs(os.path.join(cartella_origine, 'figure'), exist_ok=True)
os.makedirs(os.path.join(cartella_origine, 'tabelle'), exist_ok=True)
os.makedirs(os.path.join(cartella_origine, 'log'), exist_ok=True)
os.makedirs(os.path.join(cartella_origine, 'database'), exist_ok=True)


# Elenca tutti i file nella cartella di origine
files = os.listdir(cartella_origine)

# Sposta i file nelle cartelle corrispondenti
for file in files:
    if file.endswith('.png') or file.endswith('.jpg'):
        shutil.move(os.path.join(cartella_origine, file), os.path.join(cartella_origine, 'figure', file))
    elif file.endswith('.xlsx'):
        shutil.move(os.path.join(cartella_origine, file), os.path.join(cartella_origine, 'tabelle', file))
    elif file.endswith('.txt'):
        shutil.move(os.path.join(cartella_origine, file), os.path.join(cartella_origine, 'log', file))
    elif file.endswith('.db'):
        shutil.move(os.path.join(cartella_origine, file), os.path.join(cartella_origine, 'database', file))





end_time = datetime.now()
print('Elapsed Time: {}'.format(end_time - start_time))
plt.show()