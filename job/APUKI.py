from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QComboBox, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget, QDateTimeEdit, QFileDialog
)
from PyQt6.QtCore import QDateTime, Qt
from PyQt6.QtGui import QDoubleValidator
import sys
import easygui
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from astropy.time import Time
from astropy.coordinates import get_sun, ITRS
from astropy.utils import iers
import concurrent.futures

iers.conf.auto_download = False
iers.conf.auto_max_age = None

# Глобальный DataFrame
df = None

# Потоковый расчёт Alpha
def calc_alpha(dt):
    t = Time(dt - pd.Timedelta('03:00:00'))
    lambda_deg = 86.5
    sun_gcrs = get_sun(t)
    sun_itrs = sun_gcrs.transform_to(ITRS(obstime=t))
    s_vec = np.array([sun_itrs.x.value, sun_itrs.y.value, sun_itrs.z.value])
    s_hat = s_vec / np.linalg.norm(s_vec)
    lam = np.deg2rad(lambda_deg)
    r_hat = np.array([np.cos(lam), np.sin(lam), 0.0])
    k = np.array([0.0, 0.0, 1.0])
    n_north = k - np.dot(k, r_hat) * r_hat
    n_north /= np.linalg.norm(n_north)
    alpha_north = np.degrees(np.arccos(np.clip(np.dot(n_north, s_hat), -1.0, 1.0)))
    return 90 - alpha_north

def plan():
    global df
    plan_files = easygui.fileopenbox(filetypes="*.txt", multiple=True)
    if not plan_files:
        return

    plan_dictMain = {
        'DateTime': [], 'Direction': [], 'N': [], 'BurnTime': [],
        '2Stage': [], 'Time': [], 'Alpha': [], 'Date': [],
        '+Y': [], '+Z': [], '-Y': [], '-Z': []
    }

    txt = pd.read_csv(plan_files[0], sep='\t', header=2, encoding='ANSI', low_memory=False).drop([0])
    txt = txt[txt.columns[0]].str.split(expand=True)

    start = txt[txt[txt.columns[0]] == 'Время'].index.values[0]
    end = txt[txt[txt.columns[0]] == 'Включения'].index.values[1] - 1

    for x in range(start, end):
        date = txt.iloc[x, 0]
        time = txt.iloc[x, 1]
        dt = pd.to_datetime(date + ' ' + time, dayfirst=True)
        plan_dictMain['DateTime'].append(dt)
        plan_dictMain['Date'].append(str(dt).split(' ')[0])
        plan_dictMain['Direction'].append(txt.iloc[x, 10])
        plan_dictMain['2Stage'].append(int(txt.iloc[x, 4]))
        plan_dictMain['BurnTime'].append(float(txt.iloc[x, 2]))

    plan_dictMain['Direction'] = [
        '+Y' if x == 'Восток' else '+Z' if x == 'Юг' else '-Y' if x == 'Запад' else '-Z'
        for x in plan_dictMain['Direction']
    ]
    for key in ['+Y', '+Z', '-Y', '-Z']:
        plan_dictMain[key] = [1 if x == key else 0 for x in plan_dictMain['Direction']]
    plan_dictMain['Time'] = [x.hour*3600 + x.minute*60 + x.second for x in plan_dictMain['DateTime']]

    # N
    count = txt.iloc[txt[txt[1]=='NТМ'].index[0]:].drop(columns=txt.columns[7:]).reset_index()
    for i in range(len(count[0])):
        if i < len(count[0])-2:
            if count[0].iloc[i+1] == '001':
                plan_dictMain['N'].append(int(count[0].iloc[i]))
    plan_dictMain['N'].append(int(count[0].iloc[i]))

    # Alpha (многопоточно)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        plan_dictMain['Alpha'] = list(executor.map(calc_alpha, plan_dictMain['DateTime']))

    dfAs = pd.read_excel('AsK2.xlsx', sheet_name='As')
    dfAs['Date'] = [str(x).split(' ')[0] for x in dfAs['Date']]
    df = pd.merge(pd.DataFrame(plan_dictMain), dfAs, how='left', on='Date')

    # ---- APU модель ----
    loaded_mlp_model = joblib.load("modelApu.pkl")
    loaded_scaler = joblib.load("scalerApu.pkl")
    X_new = df[['Time', 'Alpha', 'As']]
    X_scaled = X_new.copy()
    X_scaled[['Time','Alpha','As']] = loaded_scaler.transform(X_new[['Time','Alpha','As']])
    y_pred = loaded_mlp_model.predict(X_scaled)
    df['TAPU1_0'] = [round(x+0.8,2) for x in y_pred.T[0]]
    df['TAPU2_0'] = [round(x+1,2) for x in y_pred.T[1]]

    # ---- MAX модель ----
    rf_model = joblib.load('RandomForest')
    prediction = rf_model.predict(df[['TAPU1_0','TAPU2_0','N','BurnTime','2Stage','+Y','+Z','-Y','-Z']].to_numpy())
    df['TAPU1_max'] = [round(x+y,2) for x,y in zip(df['TAPU1_0'], prediction.T[0])]
    df['TAPU2_max'] = [round(x+y,2) for x,y in zip(df['TAPU2_0'], prediction.T[1])]

class TapuPredictor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ТАПУ Прогноз")
        self.setGeometry(100,100,1350,700)

        vbox = QVBoxLayout(self)
        hbox = QHBoxLayout()

        self.load_btn = QPushButton("Загрузка Плана")
        self.load_btn.clicked.connect(self.load_plan)
        hbox.addWidget(self.load_btn)

        self.predict_btn = QPushButton("Прогноз")
        self.predict_btn.clicked.connect(self.predict_table)
        hbox.addWidget(self.predict_btn)

        self.save_btn = QPushButton("Сохранить")
        self.save_btn.clicked.connect(self.save_to_excel)
        hbox.addWidget(self.save_btn)

        vbox.addLayout(hbox)
        self.table = QTableWidget()
        vbox.addWidget(self.table)

    def load_plan(self):
        plan()
        if df is None:
            return

        self.table.setRowCount(len(df))
        self.table.setColumnCount(9)
        headers = ["Дата и Время","ТАПУ1_0","ТАПУ2_0","Кол-во вкл.","Огневое время",
                   "2-й этап","Направление","ТАПУ1_max","ТАПУ2_max"]
        self.table.setHorizontalHeaderLabels(headers)

        for i, row in df.iterrows():
            dt_edit = QDateTimeEdit(QDateTime(row['DateTime'].year, row['DateTime'].month,
                                              row['DateTime'].day, row['DateTime'].hour,
                                              row['DateTime'].minute, row['DateTime'].second))
            dt_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            dt_edit.setCalendarPopup(True)
            self.table.setCellWidget(i,0,dt_edit)

            # TAPU1_0
            le1 = QLineEdit(str(row['TAPU1_0'])); le1.setValidator(QDoubleValidator())
            self.table.setCellWidget(i,1,le1)
            # TAPU2_0
            le2 = QLineEdit(str(row['TAPU2_0'])); le2.setValidator(QDoubleValidator())
            self.table.setCellWidget(i,2,le2)
            # Кол-во вкл.
            le3 = QLineEdit(str(row['N'])); le3.setValidator(QDoubleValidator())
            self.table.setCellWidget(i,3,le3)
            # Огневое время
            le4 = QLineEdit(str(row['BurnTime'])); le4.setValidator(QDoubleValidator())
            self.table.setCellWidget(i,4,le4)
            # 2-й этап
            cb_stage = QComboBox(); cb_stage.addItems(["Есть","Нет"])
            cb_stage.setCurrentText("Есть" if row['2Stage']==1 else "Нет")
            self.table.setCellWidget(i,5,cb_stage)
            # Направление
            cb_dir = QComboBox(); cb_dir.addItems(["+Y","+Z","-Y","-Z"])
            cb_dir.setCurrentText(row['Direction'])
            self.table.setCellWidget(i,6,cb_dir)
            # TAPU1_max
            le5 = QLineEdit(str(row['TAPU1_max'])); le5.setValidator(QDoubleValidator())
            self.table.setCellWidget(i,7,le5)
            # TAPU2_max
            le6 = QLineEdit(str(row['TAPU2_max'])); le6.setValidator(QDoubleValidator())
            self.table.setCellWidget(i,8,le6)

        self.color_cells()

    def color_cells(self):
        rows = self.table.rowCount()
        for i in range(rows):
            for col in [7,8]:
                le = self.table.cellWidget(i,col)
                val = float(le.text())
                col_style = 'lightgreen' if val<39 else 'orange' if val<40 else 'red'
                le.setStyleSheet(f"background-color: {col_style}; font-weight: bold")

    def predict_table(self):
        rows = self.table.rowCount()
        if df is None or rows==0:
            return

        # Перерасчёт TAPU1_0 и TAPU2_0
        loaded_mlp_model = joblib.load("modelApu.pkl")
        loaded_scaler = joblib.load("scalerApu.pkl")
        df_tmp = pd.DataFrame(columns=['Time','Alpha','As'])
        for i in range(rows):
            dt_edit = self.table.cellWidget(i,0)
            dt = dt_edit.dateTime().toPyDateTime()  # теперь datetime с временем
            t_sec = dt.hour*3600 + dt.minute*60 + dt.second
            df_tmp.loc[i,'Time'] = t_sec
            df_tmp.loc[i,'Alpha'] = calc_alpha(pd.Timestamp(dt))
            df_tmp.loc[i,'As'] = df['As'].iloc[0]

        X_scaled = df_tmp.copy()
        X_scaled[['Time','Alpha','As']] = loaded_scaler.transform(df_tmp[['Time','Alpha','As']])
        y_pred = loaded_mlp_model.predict(X_scaled)

        for i in range(rows):
            self.table.cellWidget(i,1).setText(str(round(y_pred[i,0]+0.8,2)))
            self.table.cellWidget(i,2).setText(str(round(y_pred[i,1]+1,2)))

        # MAX модель
        rf_model = joblib.load('RandomForest')
        data_max = []
        for i in range(rows):
            t1 = float(self.table.cellWidget(i,1).text())
            t2 = float(self.table.cellWidget(i,2).text())
            cnt = float(self.table.cellWidget(i,3).text())
            bt = float(self.table.cellWidget(i,4).text())
            stage = 1 if self.table.cellWidget(i,5).currentText()=="Есть" else 0
            dir_txt = self.table.cellWidget(i,6).currentText()
            dv = {"+Y":[1,0,0,0],"+Z":[0,1,0,0],"-Y":[0,0,1,0],"-Z":[0,0,0,1]}[dir_txt]
            data_max.append([t1,t2,cnt,bt,stage,*dv])
        pred_max = rf_model.predict(np.array(data_max))
        for i in range(rows):
            t1 = float(self.table.cellWidget(i,1).text()) + pred_max[i,0]
            t2 = float(self.table.cellWidget(i,2).text()) + pred_max[i,1]
            self.table.cellWidget(i,7).setText(str(round(t1,2)))
            self.table.cellWidget(i,8).setText(str(round(t2,2)))
        self.color_cells()

    def save_to_excel(self):
        rows=self.table.rowCount(); cols=self.table.columnCount()
        headers=[self.table.horizontalHeaderItem(c).text() for c in range(cols)]
        data=[]
        for i in range(rows):
            row=[]
            for c in range(cols):
                w=self.table.cellWidget(i,c)
                if isinstance(w,QLineEdit): row.append(w.text())
                elif isinstance(w,QDateTimeEdit): row.append(w.dateTime().toString("yyyy-MM-dd HH:mm:ss"))
                elif isinstance(w,QComboBox): row.append(w.currentText())
                else:
                    it=self.table.item(i,c); row.append(it.text() if it else "")
            data.append(row)
        df_out=pd.DataFrame(data,columns=headers)
        path,_=QFileDialog.getSaveFileName(self,"Сохранить в Excel","","Excel Files (*.xlsx)")
        if path:
            if not path.endswith('.xlsx'): path+='.xlsx'
            df_out.to_excel(path,index=False)

if __name__ == "__main__":
    app=QApplication(sys.argv)
    w=TapuPredictor()
    w.show()
    sys.exit(app.exec())