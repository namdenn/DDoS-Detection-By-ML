# Data Source: Kaggle

# # Testing:
# DNS-testing.csv
# LDAP-testing.csv *
# MSSQL-testing.csv *
# NetBIOS-testing.csv *
# NTP-testing.csv
# SNMP-testing.csv
# Syn-testing.csv *
# TFTP-testing.csv
# UDPLag-testing.csv *
# UDP-testing.csv *

# # Training

# LDAP-training.csv *
# MSSQL-training.csv *
# NetBIOS-training.csv *

# Portmap-training.csv
# Syn-training.csv *

# UDPLag-training.csv *
# UDP-training.csv *

# Initialize input paths

# # LDAP
# export TRAINING_INPUT_FILE="/home/vinhngba2704/Downloads/LDAP-training.csv"
# export TESTING_INPUT_FILE="/home/vinhngba2704/Downloads/LDAP-testing.csv"

# # MSSQL
# export TRAINING_INPUT_FILE="/home/vinhngba2704/Downloads/MSSQL-training.csv"
# export TESTING_INPUT_FILE="/home/vinhngba2704/Downloads/MSSQL-testing.csv"

# # NetBIOS
# export TRAINING_INPUT_FILE="/home/vinhngba2704/Downloads/NetBIOS-training.csv"
# export TESTING_INPUT_FILE="/home/vinhngba2704/Downloads/NetBIOS-testing.csv"

# # Syn
# export TRAINING_INPUT_FILE="/home/vinhngba2704/Downloads/Syn-training.csv"
# export TESTING_INPUT_FILE="/home/vinhngba2704/Downloads/Syn-testing.csv"

# # UDP
# export TRAINING_INPUT_FILE="/home/vinhngba2704/Downloads/UDP-training.csv"
# export TESTING_INPUT_FILE="/home/vinhngba2704/Downloads/UDP-testing.csv"

# UDPLag
export TRAINING_INPUT_FILE="/home/vinhngba2704/Downloads/UDPLag-training.csv"
export TESTING_INPUT_FILE="/home/vinhngba2704/Downloads/UDPLag-testing.csv"

# Convert from >80 features to 7 basic features
python over80_to_7basic_convert.py 

# Convert from 7 basic features to 31 features
python 7basic_to_31_convert.py

# Model building and evaluation

# # Logistic Regression
# python Logistic_Regression.py

# # SVM
# python SVM.py

# Random Forest
python RandomForest.py

