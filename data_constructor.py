"""
Dataset constructor
"""

"""
Imports
"""
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from address import dataset_dir
import numpy as np
import pandas as pd
import copy

class Dataset:
    """
    Class corresponding to the dataset and several properties related
    """
    def __init__(self, dataset_name, train_fraction, seed, step) -> None:
        self.seed = seed
        self.train_fraction = train_fraction
        self.name = dataset_name
        self.step = step
        self.raw, self.binary, self.categorical, self.ordinal, self.continuous, self.label_name = self.load_file()
        self.features = self.binary + self.categorical + self.ordinal + self.continuous
        self.balanced, self.balanced_label = self.balance_data()
        self.train_pd, self.test_pd, self.train_target, self.test_target = train_test_split(self.balanced,self.balanced_label,train_size=self.train_fraction,random_state=self.seed)
        self.bin_encoder, self.cat_encoder, self.ord_scaler, self.con_scaler = self.encoder_scaler_fit()
        self.processed_train_pd, self.processed_test_pd = self.encoder_scaler_transform()
        self.features = self.train_pd.columns.to_list()
        self.bin_enc_cols = list(self.bin_encoder.get_feature_names_out(self.binary))
        self.cat_enc_cols = list(self.cat_encoder.get_feature_names_out(self.categorical))
        self.processed_features = self.bin_enc_cols + self.cat_enc_cols + self.ordinal + self.continuous
        self.feature_dist, self.processed_feat_dist = self.feature_distribution()
        self.feat_type = self.define_feat_type()
        self.feat_step = self.define_feat_step()

    def erase_missing(self,data):
        """
        Function that eliminates instances with missing values
        Output data: Filtered dataset without points with missing values
        """
        data = data.replace({'?':np.nan})
        data = data.replace({' ?':np.nan})
        if self.name == 'compass':
            for i in data.columns:
                if data[i].dtype == 'O' or data[i].dtype == 'str':
                    if len(data[i].apply(type).unique()) > 1:
                        data[i] = data[i].apply(float)
                        data.fillna(0,inplace=True)    
                    data.fillna('0',inplace=True)
                else:
                    data.fillna(0,inplace=True)
        data.dropna(axis=0,how='any',inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def nom_to_num(self,data):
        """
        Function to transform categorical features into encoded continuous values.
        Input data: The dataset to encode the categorical features.
        Output data: The dataset with categorical features encoded into continuous features.
        """
        encoder = LabelEncoder()
        if data['label'].dtypes == object or data['label'].dtypes == str:
            encoder.fit(data['label'])
            data['label'] = encoder.transform(data['label'])
        return data, encoder

    def load_file(self):
        """
        The UCI and Propublica datasets are preprocessed according to the MACE algorithm. Please, see: https://github.com/amirhk/mace.
        Function to load self.name file
        Output raw_df: Pandas DataFrame with raw data from read file
        Output binary: The binary columns names
        Output categorical: The categorical columns names
        Output ordinal: The ordinal columns names
        Output continuous: The continuous columns names
        Output label: Name of the column label
        """
        file_path = dataset_dir+self.name+'/'+self.name+'.csv'
        label = None
        if self.name == 'adult':
            binary = [] #'Sex','NativeCountry'
            categorical = ['WorkClass','Occupation','Relationship'] #'MaritalStatus'
            continuous = ['Age','EducationNumber','CapitalGain','CapitalLoss','HoursPerWeek']
            ordinal = ['EducationLevel']
            label = ['label']
            processed_df = pd.read_csv(dataset_dir+'adult/processed_adult.csv',index_col=0)
        elif self.name == 'bank':
            binary = ['Default','Housing','Loan']
            categorical = ['Job','MaritalStatus','Education','Poutcome'] # 'Contact'
            continuous = ['Balance','Duration','Campaign','Pdays','Previous'] # 'Day'
            ordinal = ['AgeGroup']
            label = ['Subscribed']
            cols = binary + categorical + ordinal + continuous + label
            processed_df = pd.read_csv(dataset_dir+'bank/bank.csv',sep=';',index_col=False)
            processed_df.loc[processed_df['age'] < 25,'AgeGroup'] = 1
            processed_df.loc[(processed_df['age'] <= 60) & (processed_df['age'] >= 25),'AgeGroup'] = 2
            processed_df.loc[processed_df['age'] > 60,'AgeGroup'] = 3
            processed_df.loc[processed_df['default'] == 'no','Default'] = 1
            processed_df.loc[processed_df['default'] == 'yes','Default'] = 2
            processed_df.loc[processed_df['housing'] == 'no','Housing'] = 1
            processed_df.loc[processed_df['housing'] == 'yes','Housing'] = 2
            processed_df.loc[processed_df['loan'] == 'no','Loan'] = 1
            processed_df.loc[processed_df['loan'] == 'yes','Loan'] = 2
            processed_df.loc[processed_df['job'] == 'management','Job'] = 1
            processed_df.loc[processed_df['job'] == 'technician','Job'] = 2
            processed_df.loc[processed_df['job'] == 'entrepreneur','Job'] = 3
            processed_df.loc[processed_df['job'] == 'blue-collar','Job'] = 4
            processed_df.loc[processed_df['job'] == 'retired','Job'] = 5
            processed_df.loc[processed_df['job'] == 'admin.','Job'] = 6
            processed_df.loc[processed_df['job'] == 'services','Job'] = 7
            processed_df.loc[processed_df['job'] == 'self-employed','Job'] = 8
            processed_df.loc[processed_df['job'] == 'unemployed','Job'] = 9
            processed_df.loc[processed_df['job'] == 'housemaid','Job'] = 10
            processed_df.loc[processed_df['job'] == 'student','Job'] = 11
            processed_df.loc[processed_df['job'] == 'unknown','Job'] = 12
            processed_df.loc[processed_df['marital'] == 'married','MaritalStatus'] = 1
            processed_df.loc[processed_df['marital'] == 'single','MaritalStatus'] = 2
            processed_df.loc[processed_df['marital'] == 'divorced','MaritalStatus'] = 3
            processed_df.loc[processed_df['education'] == 'primary','Education'] = 1
            processed_df.loc[processed_df['education'] == 'secondary','Education'] = 2
            processed_df.loc[processed_df['education'] == 'tertiary','Education'] = 3
            processed_df.loc[processed_df['education'] == 'unknown','Education'] = 4
            processed_df.loc[processed_df['contact'] == 'telephone','Contact'] = 1
            processed_df.loc[processed_df['contact'] == 'cellular','Contact'] = 2
            processed_df.loc[processed_df['contact'] == 'unknown','Contact'] = 3
            processed_df.loc[processed_df['month'] == 'jan','Month'] = 1
            processed_df.loc[processed_df['month'] == 'feb','Month'] = 2
            processed_df.loc[processed_df['month'] == 'mar','Month'] = 3
            processed_df.loc[processed_df['month'] == 'apr','Month'] = 4
            processed_df.loc[processed_df['month'] == 'may','Month'] = 5
            processed_df.loc[processed_df['month'] == 'jun','Month'] = 6
            processed_df.loc[processed_df['month'] == 'jul','Month'] = 7
            processed_df.loc[processed_df['month'] == 'ago','Month'] = 8
            processed_df.loc[processed_df['month'] == 'sep','Month'] = 9
            processed_df.loc[processed_df['month'] == 'oct','Month'] = 10
            processed_df.loc[processed_df['month'] == 'nov','Month'] = 11
            processed_df.loc[processed_df['month'] == 'dec','Month'] = 12
            processed_df.loc[processed_df['poutcome'] == 'success','Poutcome'] = 1
            processed_df.loc[processed_df['poutcome'] == 'failure','Poutcome'] = 2
            processed_df.loc[processed_df['poutcome'] == 'other','Poutcome'] = 3
            processed_df.loc[processed_df['poutcome'] == 'unknown','Poutcome'] = 4
            processed_df.loc[processed_df['y'] == 'no','Subscribed'] = int(0)
            processed_df.loc[processed_df['y'] == 'yes','Subscribed'] = int(1)
            processed_df.rename({'balance':'Balance','day':'Day','duration':'Duration','campaign':'Campaign','pdays':'Pdays','previous':'Previous'}, inplace=True, axis=1)
            processed_df = processed_df[cols]
            processed_df['Subscribed']=processed_df['Subscribed'].astype('int')
        elif self.name == 'compass':
            processed_df = pd.DataFrame()
            binary = ['Race','Sex','ChargeDegree']
            categorical = []
            continuous = ['PriorsCount']
            ordinal = ['AgeGroup']
            label = ['TwoYearRecid (label)']
            processed_df = pd.read_csv(dataset_dir+'compass/processed_compass.csv',index_col=0)
        elif self.name == 'credit':
            binary = ['isMale','isMarried','HasHistoryOfOverduePayments']
            categorical = []
            continuous = ['MaxBillAmountOverLast6Months','MaxPaymentAmountOverLast6Months','MonthsWithZeroBalanceOverLast6Months',
                    'MonthsWithLowSpendingOverLast6Months','MonthsWithHighSpendingOverLast6Months','MostRecentBillAmount',
                    'MostRecentPaymentAmount','TotalOverdueCounts','TotalMonthsOverdue']
            ordinal = ['AgeGroup','EducationLevel']
            label = ['NoDefaultNextMonth (label)']
            processed_df = pd.read_csv(dataset_dir + '/credit/credit_processed.csv') # File obtained from MACE algorithm Datasets (please, see: https://github.com/amirhk/mace)
        elif self.name == 'diabetes':
            binary = [] #'DiabetesMed'
            categorical = ['A1CResult','Metformin','Chlorpropamide','Glipizide'] #'Sex','Race','Miglitol','Acarbose','Rosiglitazone'
            continuous = ['TimeInHospital','NumProcedures','NumMedications','NumEmergency']
            ordinal = ['AgeGroup']
            label = ['Label']
            cols = binary + categorical + ordinal + continuous + label
            raw_df = pd.read_csv(dataset_dir+'diabetes/diabetes.csv') # Requires numeric transform
            cols_to_delete = ['encounter_id','patient_nbr','weight','payer_code','medical_specialty',
                            'diag_1','diag_2','diag_3','max_glu_serum','repaglinide',
                            'nateglinide','acetohexamide','glyburide','tolbutamide','pioglitazone',
                            'troglitazone','tolazamide','examide','citoglipton','insulin',
                            'glyburide-metformin','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone',
                            'change','admission_type_id','discharge_disposition_id','admission_source_id','num_lab_procedures',
                            'number_outpatient','number_inpatient','number_diagnoses']
            raw_df.drop(cols_to_delete, inplace=True, axis=1)
            raw_df = self.erase_missing(raw_df)
            raw_df = raw_df[raw_df['readmitted'] != 'NO']
            processed_df = pd.DataFrame(index=raw_df.index)
            processed_df.loc[raw_df['race'] == 'Caucasian','Race'] = 1
            processed_df.loc[raw_df['race'] == 'AfricanAmerican','Race'] = 2
            processed_df.loc[raw_df['race'] == 'Hispanic','Race'] = 3
            processed_df.loc[raw_df['race'] == 'Asian','Race'] = 4
            processed_df.loc[raw_df['race'] == 'Other','Race'] = 5
            processed_df.loc[raw_df['gender'] == 'Male','Sex'] = 1
            processed_df.loc[raw_df['gender'] == 'Female','Sex'] = 2
            processed_df.loc[(raw_df['age'] == '[0-10)') | (raw_df['age'] == '[10-20)'),'AgeGroup'] = 1
            processed_df.loc[(raw_df['age'] == '[20-30)') | (raw_df['age'] == '[30-40)'),'AgeGroup'] = 2
            processed_df.loc[(raw_df['age'] == '[40-50)') | (raw_df['age'] == '[50-60)'),'AgeGroup'] = 3
            processed_df.loc[(raw_df['age'] == '[60-70)') | (raw_df['age'] == '[70-80)'),'AgeGroup'] = 4
            processed_df.loc[(raw_df['age'] == '[80-90)') | (raw_df['age'] == '[90-100)'),'AgeGroup'] = 5
            processed_df.loc[raw_df['A1Cresult'] == 'None','A1CResult'] = 1
            processed_df.loc[raw_df['A1Cresult'] == '>7','A1CResult'] = 2
            processed_df.loc[raw_df['A1Cresult'] == 'Norm','A1CResult'] = 3
            processed_df.loc[raw_df['A1Cresult'] == '>8','A1CResult'] = 4
            processed_df.loc[raw_df['metformin'] == 'No','Metformin'] = 1
            processed_df.loc[raw_df['metformin'] == 'Steady','Metformin'] = 2
            processed_df.loc[raw_df['metformin'] == 'Up','Metformin'] = 3
            processed_df.loc[raw_df['metformin'] == 'Down','Metformin'] = 4
            processed_df.loc[raw_df['chlorpropamide'] == 'No','Chlorpropamide'] = 1
            processed_df.loc[raw_df['chlorpropamide'] == 'Steady','Chlorpropamide'] = 2
            processed_df.loc[raw_df['chlorpropamide'] == 'Up','Chlorpropamide'] = 3
            processed_df.loc[raw_df['chlorpropamide'] == 'Down','Chlorpropamide'] = 4
            processed_df.loc[raw_df['glipizide'] == 'No','Glipizide'] = 1
            processed_df.loc[raw_df['glipizide'] == 'Steady','Glipizide'] = 2
            processed_df.loc[raw_df['glipizide'] == 'Up','Glipizide'] = 3
            processed_df.loc[raw_df['glipizide'] == 'Down','Glipizide'] = 4
            processed_df.loc[raw_df['rosiglitazone'] == 'No','Rosiglitazone'] = 1
            processed_df.loc[raw_df['rosiglitazone'] == 'Steady','Rosiglitazone'] = 2
            processed_df.loc[raw_df['rosiglitazone'] == 'Up','Rosiglitazone'] = 3
            processed_df.loc[raw_df['rosiglitazone'] == 'Down','Rosiglitazone'] = 4
            processed_df.loc[raw_df['acarbose'] == 'No','Acarbose'] = 1
            processed_df.loc[raw_df['acarbose'] == 'Steady','Acarbose'] = 2
            processed_df.loc[raw_df['acarbose'] == 'Up','Acarbose'] = 3
            processed_df.loc[raw_df['acarbose'] == 'Down','Acarbose'] = 4
            processed_df.loc[raw_df['miglitol'] == 'No','Miglitol'] = 1
            processed_df.loc[raw_df['miglitol'] == 'Steady','Miglitol'] = 2
            processed_df.loc[raw_df['miglitol'] == 'Up','Miglitol'] = 3
            processed_df.loc[raw_df['miglitol'] == 'Down','Miglitol'] = 4
            processed_df.loc[raw_df['diabetesMed'] == 'No','DiabetesMed'] = 0
            processed_df.loc[raw_df['diabetesMed'] == 'Yes','DiabetesMed'] = 1
            processed_df['TimeInHospital'] = raw_df['time_in_hospital']
            processed_df['NumProcedures'] = raw_df['num_procedures']
            processed_df['NumMedications'] = raw_df['num_medications']
            processed_df['NumEmergency'] = raw_df['number_emergency']
            processed_df.loc[raw_df['readmitted'] == '<30','Label'] = 0
            processed_df.loc[raw_df['readmitted'] == '>30','Label'] = 1
            processed_df = processed_df[cols]
        elif self.name == 'dutch':
            binary = ['Sex']
            categorical = ['HouseholdPosition','HouseholdSize','Country','EconomicStatus','CurEcoActivity','MaritalStatus']
            continuous = ['Age']
            ordinal = ['EducationLevel']
            label = ['Occupation']
            cols = binary + categorical + ordinal + continuous + label
            raw_df = pd.read_csv(dataset_dir+'/dutch/dutch.txt')
            processed_df = raw_df[cols]
            processed_df.loc[processed_df['HouseholdPosition'] == 1131,'HouseholdPosition'] = 1
            processed_df.loc[processed_df['HouseholdPosition'] == 1122,'HouseholdPosition'] = 2
            processed_df.loc[processed_df['HouseholdPosition'] == 1121,'HouseholdPosition'] = 3
            processed_df.loc[processed_df['HouseholdPosition'] == 1110,'HouseholdPosition'] = 4
            processed_df.loc[processed_df['HouseholdPosition'] == 1210,'HouseholdPosition'] = 5
            processed_df.loc[processed_df['HouseholdPosition'] == 1132,'HouseholdPosition'] = 6
            processed_df.loc[processed_df['HouseholdPosition'] == 1140,'HouseholdPosition'] = 7
            processed_df.loc[processed_df['HouseholdPosition'] == 1220,'HouseholdPosition'] = 8
            processed_df.loc[processed_df['HouseholdSize'] == 111,'HouseholdSize'] = 1
            processed_df.loc[processed_df['HouseholdSize'] == 112,'HouseholdSize'] = 2
            processed_df.loc[processed_df['HouseholdSize'] == 113,'HouseholdSize'] = 3
            processed_df.loc[processed_df['HouseholdSize'] == 114,'HouseholdSize'] = 4
            processed_df.loc[processed_df['HouseholdSize'] == 125,'HouseholdSize'] = 5
            processed_df.loc[processed_df['HouseholdSize'] == 126,'HouseholdSize'] = 6
            processed_df.loc[processed_df['EconomicStatus'] == 111,'EconomicStatus'] = 1
            processed_df.loc[processed_df['EconomicStatus'] == 120,'EconomicStatus'] = 2
            processed_df.loc[processed_df['EconomicStatus'] == 112,'EconomicStatus'] = 3
            processed_df.loc[processed_df['CurEcoActivity'] == 131,'CurEcoActivity'] = 1
            processed_df.loc[processed_df['CurEcoActivity'] == 135,'CurEcoActivity'] = 2
            processed_df.loc[processed_df['CurEcoActivity'] == 138,'CurEcoActivity'] = 3
            processed_df.loc[processed_df['CurEcoActivity'] == 122,'CurEcoActivity'] = 4
            processed_df.loc[processed_df['CurEcoActivity'] == 137,'CurEcoActivity'] = 5
            processed_df.loc[processed_df['CurEcoActivity'] == 136,'CurEcoActivity'] = 6
            processed_df.loc[processed_df['CurEcoActivity'] == 133,'CurEcoActivity'] = 7
            processed_df.loc[processed_df['CurEcoActivity'] == 139,'CurEcoActivity'] = 8
            processed_df.loc[processed_df['CurEcoActivity'] == 132,'CurEcoActivity'] = 9
            processed_df.loc[processed_df['CurEcoActivity'] == 134,'CurEcoActivity'] = 10
            processed_df.loc[processed_df['CurEcoActivity'] == 111,'CurEcoActivity'] = 11
            processed_df.loc[processed_df['CurEcoActivity'] == 124,'CurEcoActivity'] = 12
            processed_df.loc[processed_df['Occupation'] == '5_4_9','Occupation'] = int(1)
            processed_df.loc[processed_df['Occupation'] == '2_1','Occupation'] = int(0)
            processed_df['Occupation']=processed_df['Occupation'].astype('int')
            processed_df.loc[processed_df['Age'] == 4,'Age'] = 15
            processed_df.loc[processed_df['Age'] == 5,'Age'] = 16
            processed_df.loc[processed_df['Age'] == 6,'Age'] = 18
            processed_df.loc[processed_df['Age'] == 7,'Age'] = 21
            processed_df.loc[processed_df['Age'] == 8,'Age'] = 22
            processed_df.loc[processed_df['Age'] == 9,'Age'] = 27
            processed_df.loc[processed_df['Age'] == 10,'Age'] = 32
            processed_df.loc[processed_df['Age'] == 11,'Age'] = 37
            processed_df.loc[processed_df['Age'] == 12,'Age'] = 42
            processed_df.loc[processed_df['Age'] == 13,'Age'] = 47
            processed_df.loc[processed_df['Age'] == 14,'Age'] = 52
            processed_df.loc[processed_df['Age'] == 15,'Age'] = 59
        elif self.name == 'german':
            binary = ['Sex']
            categorical = []
            ordinal = []
            continuous = ['Age','Credit','LoanDuration']
            label = 'GoodCustomer (label)'
            processed_df = pd.read_csv(dataset_dir+'/german/processed_german.csv')
        elif self.name == 'ionosphere':
            columns = [str(i) for i in range(34)]
            columns = columns + ['label']
            processed_df = pd.read_csv(dataset_dir+'/ionosphere/ionosphere.data',names=columns)
            columns = ['2','4','5','6','26','label']
            processed_df = processed_df[columns]
            processed_df, lbl_encoder = self.nom_to_num(processed_df)
            binary = []
            categorical = []
            ordinal = []
            continuous = ['2','4','5','6','26']
        elif self.name == 'kdd_census':
            binary = ['Sex','Race']
            categorical = ['Industry','Occupation']
            continuous = ['Age','WageHour','CapitalGain','CapitalLoss','Dividends','WorkWeeksYear']
            ordinal = []
            label = ['Label']
            cols = binary + categorical + ordinal + continuous + label
            read_cols = ['Age','WorkClass','IndustryDetail','OccupationDetail','Education','WageHour','Enrolled','MaritalStatus','Industry','Occupation',
                    'Race','Hispanic','Sex','Union','UnemployedReason','FullTimePartTime','CapitalGain','CapitalLoss','Dividends','Tax',
                    'RegionPrev','StatePrev','HouseDetailFamily','HouseDetailSummary','UnknownFeature','ChangeMsa','ChangeReg','MoveReg','Live1YrAgo','PrevSunbelt','NumPersonsWorkEmp',
                    'Under18Family','CountryFather','CountryMother','Country','Citizenship','OwnBusiness','VeteransAdmin','VeteransBenefits','WorkWeeksYear','Year','Label']
            train_raw_df = pd.read_csv(dataset_dir+'/kdd_census/census-income.data',index_col=False,names=read_cols)
            test_raw_df = pd.read_csv(dataset_dir+'/kdd_census/census-income.test',index_col=False,names=read_cols)
            raw_df = pd.concat((train_raw_df,test_raw_df),axis=0)
            raw_df.reset_index(drop=True, inplace=True)
            processed_df = raw_df[cols]
            processed_df.loc[processed_df['Sex'] == ' Male','Sex'] = 1
            processed_df.loc[processed_df['Sex'] == ' Female','Sex'] = 2
            processed_df.loc[processed_df['Race'] != ' White','Race'] = 'Non-white'
            processed_df.loc[processed_df['Race'] == ' White','Race'] = 1
            processed_df.loc[processed_df['Race'] == 'Non-white','Race'] = 2
            processed_df.loc[processed_df['Industry'] == ' Construction','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Entertainment','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Finance insurance and real estate','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Business and repair services','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Manufacturing-nondurable goods','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Personal services except private HH','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Manufacturing-durable goods','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Other professional services','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Mining','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Transportation','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Wholesale trade','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Public administration','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Retail trade','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Social services','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Private household services','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Communications','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Agriculture','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Forestry and fisheries','Industry'] = 'Industry'
            processed_df.loc[processed_df['Industry'] == ' Education','Industry'] = 'Education'
            processed_df.loc[processed_df['Industry'] == ' Utilities and sanitary services','Industry'] = 'Medical'
            processed_df.loc[processed_df['Industry'] == ' Hospital services','Industry'] = 'Medical'
            processed_df.loc[processed_df['Industry'] == ' Medical except hospital','Industry'] = 'Medical'
            processed_df.loc[processed_df['Industry'] == ' Armed Forces','Industry'] = 'Military'
            processed_df.loc[processed_df['Industry'] == ' Not in universe or children','Industry'] = 'Other'
            processed_df.loc[processed_df['Industry'] == 'Industry','Industry'] = 1
            processed_df.loc[processed_df['Industry'] == 'Education','Industry'] = 2
            processed_df.loc[processed_df['Industry'] == 'Medical','Industry'] = 3
            processed_df.loc[processed_df['Industry'] == 'Military','Industry'] = 4
            processed_df.loc[processed_df['Industry'] == 'Other','Industry'] = 5
            processed_df.loc[processed_df['Occupation'] == ' Precision production craft & repair','Occupation'] = 'Technician'
            processed_df.loc[processed_df['Occupation'] == ' Professional specialty','Occupation'] = 'Executive'
            processed_df.loc[processed_df['Occupation'] == ' Executive admin and managerial','Occupation'] = 'Executive'
            processed_df.loc[processed_df['Occupation'] == ' Handlers equip cleaners etc ','Occupation'] = 'Services'
            processed_df.loc[processed_df['Occupation'] == ' Adm support including clerical','Occupation'] = 'Services'
            processed_df.loc[processed_df['Occupation'] == ' Machine operators assmblrs & inspctrs','Occupation'] = 'Technician'
            processed_df.loc[processed_df['Occupation'] == ' Sales','Occupation'] = 'Executive'
            processed_df.loc[processed_df['Occupation'] == ' Private household services','Occupation'] = 'Services'
            processed_df.loc[processed_df['Occupation'] == ' Technicians and related support','Occupation'] = 'Technician'
            processed_df.loc[processed_df['Occupation'] == ' Transportation and material moving','Occupation'] = 'Services'
            processed_df.loc[processed_df['Occupation'] == ' Farming forestry and fishing','Occupation'] = 'Technician'
            processed_df.loc[processed_df['Occupation'] == ' Protective services','Occupation'] = 'Services'
            processed_df.loc[processed_df['Occupation'] == ' Other service','Occupation'] = 'Services'
            processed_df.loc[processed_df['Occupation'] == ' Armed Forces','Occupation'] = 'Military'
            processed_df.loc[processed_df['Occupation'] == ' Not in universe','Occupation'] = 'Other'
            processed_df.loc[processed_df['Occupation'] == 'Technician','Occupation'] = 1
            processed_df.loc[processed_df['Occupation'] == 'Executive','Occupation'] = 2
            processed_df.loc[processed_df['Occupation'] == 'Services','Occupation'] = 3
            processed_df.loc[processed_df['Occupation'] == 'Military','Occupation'] = 4
            processed_df.loc[processed_df['Occupation'] == 'Other','Occupation'] = 5
            processed_df.loc[processed_df['Label'] == ' - 50000.','Label'] = int(0)
            processed_df.loc[processed_df['Label'] == ' 50000+.','Label'] = int(1)
            processed_df['Label']=processed_df['Label'].astype(int)
        elif self.name == 'law':
            binary = ['WorkFullTime','Sex','Race']
            categorical = ['FamilyIncome','Tier']
            continuous = ['Decile1stYear','Decile3rdYear','LSAT','UndergradGPA','FirstYearGPA','CumulativeGPA']
            ordinal = []
            label = ['BarExam']
            cols = binary + categorical + ordinal + continuous + label
            raw_df = pd.read_csv(dataset_dir+'law/law.csv')
            raw_df = self.erase_missing(raw_df)
            processed_df = pd.DataFrame(index = raw_df.index)
            processed_df['Decile1stYear'] = raw_df['decile1b'].astype(int)
            processed_df['Decile3rdYear'] = raw_df['decile3'].astype(int)
            processed_df['LSAT'] = raw_df['lsat']
            processed_df['UndergradGPA'] = raw_df['ugpa']
            processed_df['FirstYearGPA'] = raw_df['zfygpa']
            processed_df['CumulativeGPA'] = raw_df['zgpa']
            processed_df['WorkFullTime'] = raw_df['fulltime'].astype(int)
            processed_df['FamilyIncome'] = raw_df['fam_inc'].astype(int)
            processed_df.loc[raw_df['male'] == 0.0,'Sex'] = 2
            processed_df.loc[raw_df['male'] == 1.0,'Sex'] = 1
            processed_df['Tier'] = raw_df['tier'].astype(int)
            processed_df['Race'] = raw_df['race'].astype(int)
            processed_df.loc[(raw_df['race'] == 1.0) | (raw_df['race'] == 2.0) | (raw_df['race'] == 3.0) | (raw_df['race'] == 4.0) | (raw_df['race'] == 5.0) | (raw_df['race'] == 6.0) | (raw_df['race'] == 8.0),'Race'] = 2
            processed_df.loc[raw_df['race'] == 7.0,'Race'] = 1
            processed_df['BarExam'] = raw_df['pass_bar'].astype(int)
        elif self.name == 'oulad':
            binary = ['Sex','Disability']
            categorical = ['Region','CodeModule','CodePresentation','HighestEducation','IMDBand']
            continuous = ['NumPrevAttempts','StudiedCredits']
            ordinal = ['AgeGroup']
            label = ['Grade']
            cols = binary + categorical + ordinal + continuous + label
            raw_df = pd.read_csv(dataset_dir+'oulad/oulad.csv')
            raw_df = self.erase_missing(raw_df)
            processed_df = pd.DataFrame(index = raw_df.index)
            processed_df.loc[raw_df['gender'] == 'M','Sex'] = 1
            processed_df.loc[raw_df['gender'] == 'F','Sex'] = 2
            processed_df.loc[raw_df['disability'] == 'N','Disability'] = 1
            processed_df.loc[raw_df['disability'] == 'Y','Disability'] = 2
            processed_df.loc[raw_df['region'] == 'East Anglian Region','Region'] = 1
            processed_df.loc[raw_df['region'] == 'Scotland','Region'] = 2
            processed_df.loc[raw_df['region'] == 'North Western Region','Region'] = 3
            processed_df.loc[raw_df['region'] == 'South East Region','Region'] = 4
            processed_df.loc[raw_df['region'] == 'West Midlands Region','Region'] = 5
            processed_df.loc[raw_df['region'] == 'Wales','Region'] = 6
            processed_df.loc[raw_df['region'] == 'North Region','Region'] = 7
            processed_df.loc[raw_df['region'] == 'South Region','Region'] = 8
            processed_df.loc[raw_df['region'] == 'Ireland','Region'] = 9
            processed_df.loc[raw_df['region'] == 'South West Region','Region'] = 10
            processed_df.loc[raw_df['region'] == 'East Midlands Region','Region'] = 11
            processed_df.loc[raw_df['region'] == 'Yorkshire Region','Region'] = 12
            processed_df.loc[raw_df['region'] == 'London Region','Region'] = 13
            processed_df.loc[raw_df['code_module'] == 'AAA','CodeModule'] = 1
            processed_df.loc[raw_df['code_module'] == 'BBB','CodeModule'] = 2
            processed_df.loc[raw_df['code_module'] == 'CCC','CodeModule'] = 3
            processed_df.loc[raw_df['code_module'] == 'DDD','CodeModule'] = 4
            processed_df.loc[raw_df['code_module'] == 'EEE','CodeModule'] = 5
            processed_df.loc[raw_df['code_module'] == 'FFF','CodeModule'] = 6
            processed_df.loc[raw_df['code_module'] == 'GGG','CodeModule'] = 7
            processed_df.loc[raw_df['code_presentation'] == '2013J','CodePresentation'] = 1
            processed_df.loc[raw_df['code_presentation'] == '2014J','CodePresentation'] = 2
            processed_df.loc[raw_df['code_presentation'] == '2013B','CodePresentation'] = 3
            processed_df.loc[raw_df['code_presentation'] == '2014B','CodePresentation'] = 4
            processed_df.loc[raw_df['highest_education'] == 'No Formal quals','HighestEducation'] = 1
            processed_df.loc[raw_df['highest_education'] == 'Post Graduate Qualification','HighestEducation'] = 2
            processed_df.loc[raw_df['highest_education'] == 'Lower Than A Level','HighestEducation'] = 3
            processed_df.loc[raw_df['highest_education'] == 'A Level or Equivalent','HighestEducation'] = 4
            processed_df.loc[raw_df['highest_education'] == 'HE Qualification','HighestEducation'] = 5
            processed_df.loc[(raw_df['imd_band'] == '0-10%') | (raw_df['imd_band'] == '10-20'),'IMDBand'] = 1
            processed_df.loc[(raw_df['imd_band'] == '20-30%') | (raw_df['imd_band'] == '30-40%'),'IMDBand'] = 2
            processed_df.loc[(raw_df['imd_band'] == '40-50%') | (raw_df['imd_band'] == '50-60%'),'IMDBand'] = 3
            processed_df.loc[(raw_df['imd_band'] == '60-70%') | (raw_df['imd_band'] == '70-80%'),'IMDBand'] = 4
            processed_df.loc[(raw_df['imd_band'] == '80-90%') | (raw_df['imd_band'] == '90-100%'),'IMDBand'] = 5
            processed_df.loc[raw_df['age_band'] == '0-35','AgeGroup'] = 1
            processed_df.loc[raw_df['age_band'] == '35-55','AgeGroup'] = 2
            processed_df.loc[raw_df['age_band'] == '55<=','AgeGroup'] = 3
            processed_df['NumPrevAttempts'] = raw_df['num_of_prev_attempts'].astype(int)
            processed_df['StudiedCredits'] = raw_df['studied_credits'].astype(int)
            processed_df.loc[raw_df['final_result'] == 'Fail','Grade'] = int(0)
            processed_df.loc[raw_df['final_result'] == 'Withdrawn','Grade'] = int(0)
            processed_df.loc[raw_df['final_result'] == 'Pass','Grade'] = int(1)
            processed_df.loc[raw_df['final_result'] == 'Distinction','Grade'] = int(1)
        elif self.name == 'student':
            # binary = ['School','Sex','AgeGroup','Address','FamilySize','ParentStatus','SchoolSupport','FamilySupport','ExtraPaid','ExtraActivities','Nursery','HigherEdu','Internet','Romantic']
            # categorical = ['MotherJob','FatherJob','SchoolReason']
            # continuous = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
            binary = ['School','Sex','AgeGroup','SchoolSupport','FamilySupport','Nursery','HigherEdu','Internet'] #'Address','FamilySize','ParentStatus','Romantic','ExtraPaid','ExtraActivities'
            categorical = ['MotherJob','FatherJob'] # 'SchoolReason'
            continuous = ['MotherEducation','FatherEducation','TravelTime','ClassFailures','GoOut']
            ordinal = []
            label = ['Grade']
            cols = binary + categorical + ordinal + continuous + label
            raw_df = pd.read_csv(dataset_dir+'student/student.csv',sep=';')
            processed_df = pd.DataFrame(index=raw_df.index)
            processed_df.loc[raw_df['age'] < 18,'AgeGroup'] = 1
            processed_df.loc[raw_df['age'] >= 18,'AgeGroup'] = 2
            processed_df.loc[raw_df['school'] == 'GP','School'] = 1
            processed_df.loc[raw_df['school'] == 'MS','School'] = 2
            processed_df.loc[raw_df['sex'] == 'M','Sex'] = 1
            processed_df.loc[raw_df['sex'] == 'F','Sex'] = 2
            processed_df.loc[raw_df['address'] == 'U','Address'] = 1
            processed_df.loc[raw_df['address'] == 'R','address'] = 2
            processed_df.loc[raw_df['famsize'] == 'LE3','FamilySize'] = 1
            processed_df.loc[raw_df['famsize'] == 'GT3','FamilySize'] = 2
            processed_df.loc[raw_df['Pstatus'] == 'T','ParentStatus'] = 1
            processed_df.loc[raw_df['Pstatus'] == 'A','ParentStatus'] = 2
            processed_df.loc[raw_df['schoolsup'] == 'yes','SchoolSupport'] = 1
            processed_df.loc[raw_df['schoolsup'] == 'no','SchoolSupport'] = 2
            processed_df.loc[raw_df['famsup'] == 'yes','FamilySupport'] = 1
            processed_df.loc[raw_df['famsup'] == 'no','FamilySupport'] = 2
            processed_df.loc[raw_df['paid'] == 'yes','ExtraPaid'] = 1
            processed_df.loc[raw_df['paid'] == 'no','ExtraPaid'] = 2
            processed_df.loc[raw_df['activities'] == 'yes','ExtraActivities'] = 1
            processed_df.loc[raw_df['activities'] == 'no','ExtraActivities'] = 2
            processed_df.loc[raw_df['nursery'] == 'yes','Nursery'] = 1
            processed_df.loc[raw_df['nursery'] == 'no','Nursery'] = 2
            processed_df.loc[raw_df['higher'] == 'yes','HigherEdu'] = 1
            processed_df.loc[raw_df['higher'] == 'no','HigherEdu'] = 2
            processed_df.loc[raw_df['internet'] == 'yes','Internet'] = 1
            processed_df.loc[raw_df['internet'] == 'no','Internet'] = 2
            processed_df.loc[raw_df['romantic'] == 'yes','Romantic'] = 1
            processed_df.loc[raw_df['romantic'] == 'no','Romantic'] = 2
            processed_df.loc[raw_df['Medu'] == 0,'MotherEducation'] = 1
            processed_df.loc[raw_df['Medu'] == 1,'MotherEducation'] = 2
            processed_df.loc[raw_df['Medu'] == 2,'MotherEducation'] = 3
            processed_df.loc[raw_df['Medu'] == 3,'MotherEducation'] = 4
            processed_df.loc[raw_df['Medu'] == 4,'MotherEducation'] = 5
            processed_df.loc[raw_df['Fedu'] == 0,'FatherEducation'] = 1
            processed_df.loc[raw_df['Fedu'] == 1,'FatherEducation'] = 2
            processed_df.loc[raw_df['Fedu'] == 2,'FatherEducation'] = 3
            processed_df.loc[raw_df['Fedu'] == 3,'FatherEducation'] = 4
            processed_df.loc[raw_df['Fedu'] == 4,'FatherEducation'] = 5
            processed_df.loc[raw_df['Mjob'] == 'at_home','MotherJob'] = 1
            processed_df.loc[raw_df['Mjob'] == 'health','MotherJob'] = 2
            processed_df.loc[raw_df['Mjob'] == 'services','MotherJob'] = 3
            processed_df.loc[raw_df['Mjob'] == 'teacher','MotherJob'] = 4
            processed_df.loc[raw_df['Mjob'] == 'other','MotherJob'] = 5
            processed_df.loc[raw_df['Fjob'] == 'at_home','FatherJob'] = 1
            processed_df.loc[raw_df['Fjob'] == 'health','FatherJob'] = 2
            processed_df.loc[raw_df['Fjob'] == 'services','FatherJob'] = 3
            processed_df.loc[raw_df['Fjob'] == 'teacher','FatherJob'] = 4
            processed_df.loc[raw_df['Fjob'] == 'other','FatherJob'] = 5
            processed_df.loc[raw_df['reason'] == 'course','SchoolReason'] = 1
            processed_df.loc[raw_df['reason'] == 'home','SchoolReason'] = 2
            processed_df.loc[raw_df['reason'] == 'reputation','SchoolReason'] = 3
            processed_df.loc[raw_df['reason'] == 'other','SchoolReason'] = 4
            processed_df['TravelTime'] = raw_df['traveltime'].astype('int')
            processed_df['ClassFailures'] = raw_df['failures'].astype('int')
            processed_df['GoOut'] = raw_df['goout'].astype('int')
            processed_df.loc[raw_df['G3'] < 10,'Grade'] = int(0)
            processed_df.loc[raw_df['G3'] >= 10,'Grade'] = int(1)
            processed_df = processed_df[cols]
        elif self.name == 'synthetic_diagonal_plane_1ord_2con':
            processed_df = pd.read_csv(file_path)
            columns = [str(i) for i in range(processed_df.shape[1]-1)] + ['label']
            binary = []
            categorical = []
            ordinal = ['0']
            continuous = ['1','2']
            processed_df.columns = columns
            if label is None:
                label_name = processed_df.columns[-1]
            else:
                label_name = label
        elif self.name == 'synthetic_diagonal_plane_2ord_1con':
            processed_df = pd.read_csv(file_path)
            columns = [str(i) for i in range(processed_df.shape[1]-1)] + ['label']
            binary = []
            categorical = []
            ordinal = ['0','1']
            continuous = ['2']
            processed_df.columns = columns
            if label is None:
                label_name = processed_df.columns[-1]
            else:
                label_name = label
        elif self.name == 'synthetic_diagonal_plane_1bin_1ord_1con':
            processed_df = pd.read_csv(file_path)
            columns = [str(i) for i in range(processed_df.shape[1]-1)] + ['label']
            binary = ['0']
            categorical = []
            ordinal = ['1']
            continuous = ['2']
            processed_df.columns = columns
            if label is None:
                label_name = processed_df.columns[-1]
            else:
                label_name = label
        elif self.name == 'synthetic_diagonal_plane_2bin_1con':
            processed_df = pd.read_csv(file_path)
            columns = [str(i) for i in range(processed_df.shape[1]-1)] + ['label']
            binary = ['0','1']
            categorical = []
            ordinal = []
            continuous = ['2']
            processed_df.columns = columns
            if label is None:
                label_name = processed_df.columns[-1]
            else:
                label_name = label
        elif self.name == 'synthetic_diagonal_plane_2ord_1bin':
            processed_df = pd.read_csv(file_path)
            columns = [str(i) for i in range(processed_df.shape[1]-1)] + ['label']
            binary = ['2']
            categorical = []
            ordinal = ['0','1']
            continuous = []
            processed_df.columns = columns
            if label is None:
                label_name = processed_df.columns[-1]
            else:
                label_name = label
        else:
            processed_df = pd.read_csv(file_path)
            columns = [str(i) for i in range(processed_df.shape[1]-1)] + ['label']
            binary = []
            categorical = []
            ordinal = []
            continuous = columns[:-1]
            processed_df.columns = columns
        if label is None:
            label_name = processed_df.columns[-1]
        else:
            label_name = label
        return processed_df, binary, categorical, ordinal, continuous, label_name
    
    def define_feat_type(self):
        """
        Method that obtains a feature type vector corresponding to each of the features
        """
        feat_type = copy.deepcopy(self.processed_train_pd.dtypes)
        feat_list = feat_type.index.tolist()
        if self.name == 'synthetic_circle':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_0':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_1_8':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_cubic_0':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_cubic_1_8':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_plane_1ord_2con':
            for i in range(len(feat_list)):
                if i in [0]:
                    feat_type.iloc[i] = 'ord'
                else:
                    feat_type.iloc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_plane_2ord_1con':
            for i in range(len(feat_list)):
                if i in [0,1]:
                    feat_type.iloc[i] = 'ord'
                else:
                    feat_type.iloc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_plane_1bin_1ord_1con':
            for i in range(len(feat_list)):
                if i in [0]:
                    feat_type.iloc[i] = 'bin'
                elif i in [1]:
                    feat_type.iloc[i] = 'ord'
                else:
                    feat_type.iloc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_plane_2bin_1con':
            for i in range(len(feat_list)):
                if i in [0,1]:
                    feat_type.iloc[i] = 'bin'
                else:
                    feat_type.iloc[i] = 'cont'
        elif self.name == 'synthetic_diagonal_plane_2ord_1bin':
            for i in range(len(feat_list)):
                if i in [0,1]:
                    feat_type.iloc[i] = 'ord'
                else:
                    feat_type.iloc[i] = 'bin'
        elif self.name == '3-cubic':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'sinusoid':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        elif self.name == 'adult':
            for i in feat_list:
                if 'Sex' in i or 'Native' in i or 'Race' in i:
                    feat_type.loc[i] = 'bin'
                elif 'WorkClass' in i or 'Marital' in i or 'Occupation' in i or 'Relation' in i:
                    feat_type.loc[i] = 'cat'
                elif 'EducationLevel' in i:
                    feat_type.loc[i] = 'ord'
                elif 'EducationNumber' in i or 'Age' in i or 'Capital' in i or 'Hours' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'kdd_census':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i:
                    feat_type.loc[i] = 'bin'
                elif  'Industry' in i or 'Occupation' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i or 'WageHour' in i or 'Capital' in i or 'Dividends' in i or 'WorkWeeksYear' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'german':
            for i in feat_list:
                if 'Sex' in i or 'Single' in i or 'Unemployed' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Housing' in i or 'PurposeOfLoan' in i or 'InstallmentRate' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i or 'Credit' in i or 'Loan' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'dutch':
            for i in feat_list:
                if 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'HouseholdPosition' in i or 'HouseholdSize' in i or 'Country' in i or 'EconomicStatus' in i or 'CurEcoActivity' in i or 'MaritalStatus' in i:
                    feat_type.loc[i] = 'cat'
                elif 'EducationLevel' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Age' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'bank':
            for i in feat_list:
                if 'Default' in i or 'Housing' in i or 'Loan' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Job' in i or 'MaritalStatus' in i or 'Education' in i or 'Contact' in i or 'Month' in i or 'Poutcome' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Age' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Balance' in i or 'Day' in i or 'Duration' in i or 'Campaign' in i or 'Pdays' in i or 'Previous' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'credit':
            for i in feat_list:
                if 'Male' in i or 'Married' in i or 'History' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Age' in i or 'Education' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Total' in i or 'Amount' in i or 'Balance' in i or 'Spending' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'compass':
            for i in feat_list:
                if 'Sex' in i or 'Race' in i or 'Charge' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Age' in i:
                    feat_type.loc[i] = 'ord'
                elif 'Priors' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'diabetes':
            for i in feat_list:
                if 'DiabetesMed' in i or 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Race' in i or 'A1CResult' in i or 'Metformin' in i or 'Chlorpropamide' in i or 'Glipizide' in i or 'Rosiglitazone' in i or 'Acarbose' in i or 'Miglitol' in i:
                    feat_type.loc[i] = 'cat'
                elif 'AgeGroup' in i:
                    feat_type.loc[i] = 'ord'
                elif 'TimeInHospital' in i or 'NumProcedures' in i or 'NumMedications' in i or 'NumEmergency' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'student':
            for i in feat_list:
                if 'Age' in i or 'School' in i or 'Sex' in i or 'Address' in i or 'FamilySize' in i or 'ParentStatus' in i or 'SchoolSupport' in i or 'FamilySupport' in i or 'ExtraPaid' in i or 'ExtraActivities' in i or 'Nursery' in i or 'HigherEdu' in i or 'Internet' in i or 'Romantic' in i:
                    feat_type.loc[i] = 'bin'
                elif 'MotherJob' in i or 'FatherJob' in i or 'SchoolReason' in i:
                    feat_type.loc[i] = 'cat'
                elif 'MotherEducation' in i or 'FatherEducation' in i:
                    feat_type.loc[i] = 'cont'
                elif 'TravelTime' in i or 'ClassFailures' in i or 'GoOut' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'oulad':
            for i in feat_list:
                if 'Sex' in i or 'Disability' in i:
                    feat_type.loc[i] = 'bin'
                elif 'Region' in i or 'CodeModule' in i or 'CodePresentation' in i or 'HighestEducation' in i or 'IMDBand' in i:
                    feat_type.loc[i] = 'cat'
                elif 'NumPrevAttempts' in i or 'StudiedCredits' in i:
                    feat_type.loc[i] = 'cont'
                elif 'AgeGroup' in i:
                    feat_type.loc[i] = 'ord'
        elif self.name == 'law':
            for i in feat_list:
                if 'Race' in i or 'WorkFullTime' in i or 'Sex' in i:
                    feat_type.loc[i] = 'bin'
                elif 'FamilyIncome' in i or 'Tier' in i:
                    feat_type.loc[i] = 'cat'
                elif 'Decile1stYear' in i or 'Decile3rdYear' in i or 'LSAT' in i or 'UndergradGPA' in i or 'FirstYearGPA' in i or 'CumulativeGPA' in i:
                    feat_type.loc[i] = 'cont'
        elif self.name == 'ionosphere':
            for i in feat_list:
                feat_type.loc[i] = 'cont'
        return feat_type
    
    def define_feat_step(self):
        """
        Method that estimates the step size of all features (used for ordinal features)
        """
        with_cont = 'cont' in self.feat_type.values
        with_ord = 'ord' in self.feat_type.values
        with_both = with_cont and with_ord
        if with_cont and not with_ord:
            feat_step_cont = pd.Series(data=[self.step]*len(self.continuous), index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['cont']])
            feat_step = feat_step_cont
        elif with_ord and not with_cont:
            feat_step_ord = pd.Series(data=1/(self.ord_scaler.data_max_ - self.ord_scaler.data_min_), index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['ord']])
            feat_step = feat_step_ord
        elif with_both:
            feat_step_cont = pd.Series(data=[self.step]*len(self.continuous), index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['cont']])
            feat_step_ord = pd.Series(data=1/(self.ord_scaler.data_max_ - self.ord_scaler.data_min_), index=[i for i in self.feat_type.keys() if self.feat_type[i] in ['ord']])
            feat_step = pd.concat((feat_step_ord, feat_step_cont))
        # for i in self.feat_type.keys().tolist():
        #     if self.feat_type.loc[i] == 'cont':
        #         feat_step.loc[i] = self.step
        #     elif self.feat_type.loc[i] == 'ord':
        #         continue
        #     else:
        #         feat_step.loc[i] = 0
        feat_step = feat_step.reindex(index = self.feat_type.keys().to_list())
        return feat_step

    def balance_data(self):
        """
        Method to balance the dataset using undersampling of majority class
        """
        data_label = self.raw[self.label_name]
        label_value_counts = data_label.value_counts()
        samples_per_class = label_value_counts.min()
        balanced_df = pd.concat([self.raw[(data_label == 0).to_numpy()].sample(samples_per_class, random_state = self.seed),
        self.raw[(data_label == 1).to_numpy()].sample(samples_per_class, random_state = self.seed),]).sample(frac = 1, random_state = self.seed)
        balanced_df_label = balanced_df[self.label_name]
        try:
            del balanced_df[self.label_name]
        except:
            del balanced_df[self.label_name[0]]
        return balanced_df, balanced_df_label

    def encoder_scaler_fit(self):
        """
        Method to fit encoder and scaler for the dataset
        """
        bin_train_pd, cat_train_pd, ord_train_pd, con_train_pd = self.train_pd[self.binary], self.train_pd[self.categorical], self.train_pd[self.ordinal], self.train_pd[self.continuous]
        cat_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        bin_enc = OneHotEncoder(drop='if_binary',dtype=np.uint8,handle_unknown='ignore')
        ord_scaler = MinMaxScaler(clip=True)
        con_scaler = MinMaxScaler(clip=True)
        bin_enc.fit(bin_train_pd)
        cat_enc.fit(cat_train_pd)
        if ord_train_pd.shape[1] > 0:
            ord_scaler.fit(ord_train_pd)
        if con_train_pd.shape[1] > 0:
            con_scaler.fit(con_train_pd)
        return bin_enc, cat_enc, ord_scaler, con_scaler

    def encoder_scaler_transform(self):
        """
        Method to fit encoder and scaler for the dataset
        """
        processed_train_pd = pd.DataFrame()
        bin_train_pd, cat_train_pd, ord_train_pd, con_train_pd = self.train_pd[self.binary], self.train_pd[self.categorical], self.train_pd[self.ordinal], self.train_pd[self.continuous]
        if bin_train_pd.shape[1] > 0:
            bin_enc_train_np, bin_enc_cols = self.bin_encoder.transform(bin_train_pd).toarray(), self.bin_encoder.get_feature_names_out(self.binary)
            bin_enc_train_pd = pd.DataFrame(bin_enc_train_np,index=bin_train_pd.index,columns=bin_enc_cols)
            processed_train_pd = pd.concat((processed_train_pd,bin_enc_train_pd),axis=1)
        if cat_train_pd.shape[1] > 0:
            cat_enc_train_np, cat_enc_cols = self.cat_encoder.transform(cat_train_pd).toarray(), self.cat_encoder.get_feature_names_out(self.categorical) 
            cat_enc_train_pd = pd.DataFrame(cat_enc_train_np,index=cat_train_pd.index,columns=cat_enc_cols)
            processed_train_pd = pd.concat((processed_train_pd,cat_enc_train_pd),axis=1)
        if ord_train_pd.shape[1] > 0:
            ord_scaled_train_np = self.ord_scaler.transform(ord_train_pd)
            ord_scaled_train_pd = pd.DataFrame(ord_scaled_train_np,index=ord_train_pd.index,columns=self.ordinal)
            processed_train_pd = pd.concat((processed_train_pd,ord_scaled_train_pd),axis=1)
        if con_train_pd.shape[1] > 0:
            con_scaled_train_np = self.con_scaler.transform(con_train_pd)
            con_scaled_train_pd = pd.DataFrame(con_scaled_train_np,index=con_train_pd.index,columns=self.continuous)
            processed_train_pd = pd.concat((processed_train_pd,con_scaled_train_pd),axis=1)
        
        processed_test_pd = pd.DataFrame()
        bin_test_pd, cat_test_pd, ord_test_pd, con_test_pd = self.test_pd[self.binary], self.test_pd[self.categorical], self.test_pd[self.ordinal], self.test_pd[self.continuous]
        if bin_test_pd.shape[1] > 0:
            bin_enc_test_np = self.bin_encoder.transform(bin_test_pd).toarray()
            bin_enc_test_pd = pd.DataFrame(bin_enc_test_np,index=bin_test_pd.index,columns=bin_enc_cols)
            processed_test_pd = pd.concat((processed_test_pd,bin_enc_test_pd),axis=1)
        if cat_test_pd.shape[1] > 0:
            cat_enc_test_np = self.cat_encoder.transform(cat_test_pd).toarray()
            cat_enc_test_pd = pd.DataFrame(cat_enc_test_np,index=cat_test_pd.index,columns=cat_enc_cols)
            processed_test_pd = pd.concat((processed_test_pd,cat_enc_test_pd),axis=1)
        if ord_test_pd.shape[1] > 0:
            ord_scaled_test_np = self.ord_scaler.transform(ord_test_pd)
            ord_scaled_test_pd = pd.DataFrame(ord_scaled_test_np,index=ord_test_pd.index,columns=self.ordinal)
            processed_test_pd = pd.concat((processed_test_pd,ord_scaled_test_pd),axis=1)
        if con_test_pd.shape[1] > 0:
            con_scaled_test_np = self.con_scaler.transform(con_test_pd)
            con_scaled_test_pd = pd.DataFrame(con_scaled_test_np,index=con_test_pd.index,columns=self.continuous)
            processed_test_pd = pd.concat((processed_test_pd,con_scaled_test_pd),axis=1)

        return processed_train_pd, processed_test_pd 
    
    def feature_distribution(self):
        """
        Method to calculate the distribution for all features
        """
        num_instances_balanced = self.balanced.shape[0]
        num_instances_processed_train = self.processed_train_pd.shape[0]
        feat_dist = {}
        processed_feat_dist = {}
        all_non_con_feat = self.binary+self.categorical+self.ordinal
        all_non_con_processed_feat = self.bin_enc_cols+self.cat_enc_cols+self.ordinal
        if len(all_non_con_feat) > 0:
            for i in all_non_con_feat:
                feat_dist[i] = ((self.balanced[i].value_counts()+1)/(num_instances_balanced+len(np.unique(self.balanced[i])))).to_dict() # +1 for laplacian counter
        if len(self.continuous) > 0:
            for i in self.continuous:
                feat_dist[i] = {'mean': self.balanced[i].mean(), 'std': self.balanced[i].std()}
                processed_feat_dist[i] = {'mean': self.processed_train_pd[i].mean(), 'std': self.processed_train_pd[i].std()}
        if len(all_non_con_processed_feat) > 0:
            for i in all_non_con_processed_feat:
                processed_feat_dist[i] = ((self.processed_train_pd[i].value_counts()+1)/(num_instances_processed_train+len(np.unique(self.processed_train_pd[i])))).to_dict() # +1 for laplacian counter
        return feat_dist, processed_feat_dist