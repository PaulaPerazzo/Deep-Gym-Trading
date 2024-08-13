import pandas as pd

### adjust test and validation dataset ###

validation_data = pd.read_csv("../../data/validation_data.csv")
test_data = pd.read_csv("../../data/test_data.csv")

### 1. Data Analysis ###
validation_data.drop(columns=["Unnamed: 0"], inplace=True)
test_data.drop(columns=["Unnamed: 0"], inplace=True)

validation_data.set_index("Ticker", inplace=True)
test_data.set_index("Ticker", inplace=True)

print(validation_data.describe())
print(test_data.describe())

### 2. remove unuseful columns ###
removed_columns =  ['LLIS3.SA.4', 'TIET4.SA.4', 'ITEC3.SA.4', 'VIVA3.SA.4', 'VIVT4.SA.4', 'WLMM3.SA.4', 'BTOW3.SA.4', 'VAMO3.SA.4', 'VVAR3.SA.4', 'JBDU2.SA.4', 'ELEK3.SA.4', 'RLOG3.SA.4', 'CCPR3.SA.4', 'NEOE3.SA.4', 'RNEW4.SA.4', 'CESP6.SA.4', 'JBDU3.SA.4', 'IDNT3.SA.4', 'TIMP3.SA.4', 'TCNO4.SA.4', 'LCAM3.SA.4', 'JBDU1.SA.4', 'NATU3.SA.4', 'SMLS3.SA.4', 'TESA3.SA.4', 'CESP3.SA.4', 'MEND5.SA.4', 'SEDU3.SA.4', 'IRBR3.SA.4', 'TAEE4.SA.4', 'SULA3.SA.4', 'GNDI3.SA.4', 'GFSA1.SA.4', 'CNTO3.SA.4', 'IDVL3.SA.4', 'SULA4.SA.4', 'ELEK4.SA.4', 'MTIG4.SA.4', 'CPRE3.SA.4', 'TIET2.SA.4', 'BSEV3.SA.4', 'BMGB4.SA.4', 'CEPE5.SA.4', 'IDVL4.SA.4', 'CELP5.SA.4', 'OMGE3.SA.4', 'JSLG3.SA.4', 'HAPV3.SA.4', 'TOYB3.SA.4', 'SPRI3.SA.4', 'IGTA3.SA.4', 'MMXM3.SA.4', 'PARD3.SA.4', 'LAME4.SA.4', 'CAMB4.SA.4', 'INEP4.SA.4', 'JBDU4.SA.4', 'LINX3.SA.4', 'BBRK3.SA.4', 'BTTL3.SA.4', 'WIZS3.SA.4', 'BIDI3.SA.4', 'IGBR3.SA.4', 'SQIA3.SA.4', 'MOVI3.SA.4', 'RANI4.SA.4', 'FRTA3.SA.4', 'CELP6.SA.4', 'CEAB3.SA.4', 'PCAR4.SA.4', 'AZUL4.SA.4', 'BKBR3.SA.4', 'BRML3.SA.4', 'TRPN3.SA.4', 'IDVL9.SA.4', 'BRDT3.SA.4', 'CELP7.SA.4', 'TOYB4.SA.4', 'CCXC3.SA.4', 'CELP3.SA.4', 'BIDI4.SA.4', 'ELPL3.SA.4', 'GPCP3.SA.4', 'TIET3.SA.4', 'TCNO3.SA.4', 'MEND6.SA.4', 'LIQO3.SA.4', 'DMMO3.SA.4', 'CESP5.SA.4', 'CRDE3.SA.4', 'DTEX3.SA.4', 'TCR11.SA.4', 'CAML3.SA.4', 'EEEL4.SA.4', 'PNVL4.SA.4', 'HGTX3.SA.4', 'ENBR3.SA.4', 'CRFB3.SA.4', 'LOGG3.SA.4', 'CARD3.SA.4', 'LAME3.SA.4', 'EEEL3.SA.4', 'JPSA3.SA.4', 'ALSO3.SA.4']

validation_data.drop(columns=removed_columns, inplace=True)
test_data.drop(columns=removed_columns, inplace=True)

validation_data.to_csv("../../data/validation_data_cleaned.csv")
test_data.to_csv("../../data/test_data_cleaned.csv")

### 3. Remove nan values to mean (mean train) ###
train_data = pd.read_csv("../../data/train_data_cleaned_nan.csv")

for column in validation_data:
    for line in validation_data[column]:
        if pd.isna(line):
            validation_data[column].fillna(train_data[column].mean(), inplace=True)

print(validation_data.isna().sum())
validation_data.to_csv("../../data/validation_data_cleaned_nan.csv")

for column in test_data:
    for line in test_data[column]:
        if pd.isna(line):
            test_data[column].fillna(train_data[column].mean(), inplace=True)

print(test_data.isna().sum())
test_data.to_csv("../../data/test_data_cleaned_nan.csv")
